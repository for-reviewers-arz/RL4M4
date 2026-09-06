"""
rl_mosi.py
==========
RL-BFS neighbour-selection pipeline for CMU-MOSI REGRESSION (continuous sentiment).

The policy selects the neighbour whose label is closest to the query label
(minimum absolute difference).  Final metrics include MSE and four
classification metrics derived by binarising / discretising predictions.

Three frozen pre-trained encoders produce per-modality embeddings:
  • Text  : BERT-base-uncased          → 768-dim
  • Audio : WavLM-Large                → 1024-dim
  • Video : CLIP ViT-B/32 (mean pool)  → 768-dim

All embeddings are extracted once, kept entirely in GPU VRAM, and reused
throughout Phase 2 (RL training).  No disk caching is performed.

Binary label: y = 1 if sentiment_score >= 0 else 0.
Missing-modality simulation: NONE  (complete modality config = "100_text_100_audio_100_video").

Collate strategy (Q3 answer)
─────────────────────────────
audio  → Python list of raw waveform tensors   (WavLM pads internally)
video  → Python list of OpenCV frame lists     (CLIP processes frame-by-frame)
text   → Python list of strings                (BERT tokenises in batch)
label / has_* masks → torch.stack-ed tensors

This is the only viable approach because WavLM-Large pads to the longest
waveform in a batch internally, and CLIP processes each frame individually.
Stacking variable-length tensors directly would corrupt the data.

═══════════════════════════════════════════════════════════════════════════
ALL TUNEABLE PARAMETERS ARE IN THE "PARAMETERS" BLOCK BELOW.
═══════════════════════════════════════════════════════════════════════════
"""

# ── standard library ──────────────────────────────────────────────────────────
import os
import copy
import heapq

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score

# ── pretrained encoder dependencies ───────────────────────────────────────────
from transformers import (
    BertTokenizer, BertModel,
    Wav2Vec2FeatureExtractor, WavLMModel,
    CLIPImageProcessor, CLIPVisionModel,
)
from PIL import Image

# ── project dataset ───────────────────────────────────────────────────────────
from mosi_reg import MOSIDatasetRegression

# =============================================================================
# PARAMETERS  –  edit only this block
# =============================================================================

DSS_PATH   = "/home"
AUDIO_DIR  = DSS_PATH + "/_Dataset/Raw - CMU Multimodal Opinion Sentiment Intensity/Audio/WAV_16000/Segmented"
VIDEO_DIR  = DSS_PATH + "/_Dataset/Raw - CMU Multimodal Opinion Sentiment Intensity/Video/Segmented"
TEXT_DIR   = DSS_PATH + "/_Dataset/Raw - CMU Multimodal Opinion Sentiment Intensity/Transcript/Segmented"
SPLIT_FILE = DSS_PATH + "/_Dataset/Raw - CMU Multimodal Opinion Sentiment Intensity/mosi_splits-70train.json"

MISSING_CONFIG = "20_text_100_audio_100_video"     # text 20%, audio 100%, video 20%
SEED           = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── modality dimensions (fixed by encoder architecture) ──────────────────────
TEXT_DIM  = 768    # BERT-base CLS token
AUDIO_DIM = 1024   # WavLM-Large mean-pool
VIDEO_DIM = 768    # CLIP ViT-B/32 pooler_output
N_MODS    = 3

# ── label conversion thresholds (for final classification metrics only) ────────
BINARISE_THRESHOLD = 0.0    # score >= 0 → positive (binary)
SEVEN_CLASS_BINS   = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]  # edges for 7-class

# ── video encoder ─────────────────────────────────────────────────────────────
NUM_FRAMES = 8   # frames sampled per video clip

# ── embedding extraction batch size ──────────────────────────────────────────
# One DataLoader encodes all three modalities per batch (single pass).
# Bottleneck is WavLM-Large (audio). Start at 8; increase if VRAM allows.
EXTRACT_BATCH = 8

# ── embedding variant ─────────────────────────────────────────────────────────
VARIANT = "without"   # 'with' | 'just_scaling' | 'without'

# ── RL / BFS ──────────────────────────────────────────────────────────────────
STRATEGY      = "knn"   # 'unn' | 'knn'
K_APPROX      = 20      # neighbours retrieved for UNN / seed computation
K_SEED_KNN    = 5       # k for seed set when STRATEGY='knn'
K_GRAPH       = 5       # k for the BFS adjacency graph
MAX_BFS_DEPTH = 6       # condition B: max hop depth
MAX_NODE2_RL  = 200     # condition C: max nodes collected

PPO_EPOCHS     = 10
PPO_CLIP       = 0.2
LR             = 3e-4
ENT_COEF       = 0.0001
MINIBATCH_SIZE = 128
RL_CKPT        = f"best_rl_mosi_{VARIANT}_{STRATEGY}.pt"

# ── Fusion head (attends over top-K' RL-ranked candidates) ──────────────────
FUSION_TOPK     = 32    # K': how many top-ranked candidates the head sees
FUSION_HIDDEN   = 128   # width of the cross-attention / feed-forward layers
FUSION_HEADS    = 4     # number of attention heads
FUSION_EPOCHS   = 50    # supervised regression epochs for the fusion head
FUSION_LR       = 3e-4  # Adam LR for fusion head
FUSION_BATCH    = 32    # mini-batch size for fusion head training
FUSION_CKPT     = f"best_fusion_mosi_{VARIANT}_{STRATEGY}.pt"

# =============================================================================
# FROZEN ENCODERS
# =============================================================================

class FrozenTextEncoder(nn.Module):
    """Frozen BERT encoder for text  →  (B, 768)"""
    def __init__(self):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model     = BertModel.from_pretrained('bert-base-uncased')
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def forward(self, texts):
        with torch.no_grad():
            encoding = self.tokenizer(
                texts, padding=True, truncation=True,
                max_length=128, return_tensors='pt'
            )
            encoding = {k: v.to(next(self.model.parameters()).device)
                        for k, v in encoding.items()}
            outputs  = self.model(**encoding)
            return outputs.last_hidden_state[:, 0, :]   # CLS token


class FrozenAudioEncoder(nn.Module):
    """Frozen WavLM-Large encoder for audio  →  (B, 1024)"""
    def __init__(self):
        super().__init__()
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "microsoft/wavlm-large")
        self.model = WavLMModel.from_pretrained("microsoft/wavlm-large")
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def forward(self, audio_batch):
        with torch.no_grad():
            processed = []
            for audio in audio_batch:
                if isinstance(audio, torch.Tensor):
                    if audio.shape[0] > 1:
                        audio = audio.mean(dim=0, keepdim=True)
                    waveform = audio.squeeze().cpu().numpy()
                else:
                    waveform = audio
                if waveform.ndim > 1:
                    waveform = waveform.flatten()
                inputs = self.feature_extractor(
                    waveform, sampling_rate=16000,
                    return_tensors="pt", padding=False
                )
                processed.append(inputs.input_values.squeeze(0))

            max_len = max(p.shape[-1] for p in processed)
            padded  = torch.stack([
                F.pad(p, (0, max_len - p.shape[-1])) for p in processed
            ]).to(next(self.model.parameters()).device)

            outputs = self.model(padded)
            return outputs.last_hidden_state.mean(dim=1)   # (B, 1024)


class FrozenVideoEncoder(nn.Module):
    """Frozen CLIP ViT-B/32 encoder for video  →  (B, 768)"""
    def __init__(self, num_frames: int = NUM_FRAMES):
        super().__init__()
        self.num_frames = num_frames
        self.processor  = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-base-patch32")
        self.model      = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch32")
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def forward(self, video_batch):
        device = next(self.model.parameters()).device
        with torch.no_grad():
            all_features = []
            for frames in video_batch:
                if len(frames) == 0:
                    all_features.append(
                        torch.zeros(768, device=device))
                    continue

                if len(frames) >= self.num_frames:
                    indices = np.linspace(
                        0, len(frames) - 1, self.num_frames, dtype=int)
                else:
                    repeat_factor = (
                        self.num_frames + len(frames) - 1) // len(frames)
                    indices = np.tile(
                        np.arange(len(frames)),
                        repeat_factor)[:self.num_frames]

                frame_feats = []
                for i in indices:
                    frame     = frames[i][:, :, ::-1]          # BGR→RGB
                    pil_img   = Image.fromarray(frame.astype('uint8'))
                    inputs    = self.processor(
                        images=pil_img, return_tensors="pt")
                    pv        = inputs["pixel_values"].to(device)
                    out       = self.model(pixel_values=pv)
                    frame_feats.append(out.pooler_output.squeeze(0))

                video_feat = torch.stack(frame_feats).mean(dim=0)  # (768,)
                all_features.append(video_feat)

            return torch.stack(all_features).to(device)   # (B, 768)


# =============================================================================
# COLLATE FUNCTION
# =============================================================================

def collate_fn(batch):
    """
    audio  → Python list (variable-length waveforms; WavLM pads internally)
    video  → Python list (variable-length frame lists; CLIP processes individually)
    text   → Python list of strings (BERT tokenises in batch)
    label / has_* → stacked tensors
    """
    return {
        'name':      [b['name']      for b in batch],
        'text':      [b['text']      for b in batch],
        'audio':     [b['audio']     for b in batch],
        'video':     [b['video']     for b in batch],
        'label':     torch.stack([b['label']     for b in batch]),
        'has_text':  torch.tensor([b['has_text']  for b in batch]),
        'has_audio': torch.tensor([b['has_audio'] for b in batch]),
        'has_video': torch.tensor([b['has_video'] for b in batch]),
    }


# =============================================================================
# EMBEDDING EXTRACTION  (all results kept in GPU VRAM as float32 tensors)
# =============================================================================

@torch.no_grad()
def _encode_split(dataset, text_enc, audio_enc, video_enc,
                  batch_size: int = EXTRACT_BATCH):
    """
    Single-pass extraction: one DataLoader, all three encoders called
    per batch.  Three separate passes over disk are replaced by one.

    Returns
    -------
    text_emb  : torch.Tensor  (N, TEXT_DIM)   on DEVICE
    audio_emb : torch.Tensor  (N, AUDIO_DIM)  on DEVICE
    video_emb : torch.Tensor  (N, VIDEO_DIM)  on DEVICE
    labels    : torch.Tensor  (N,) float32 continuous scores on DEVICE
    """
    N      = len(dataset)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, collate_fn=collate_fn, num_workers=0)
    n_batches = len(loader)

    text_emb_list  = []
    audio_emb_list = []
    video_emb_list = []
    label_list     = []

    for i, batch in enumerate(loader):
        print(f"\r    batch {i+1}/{n_batches}", end="", flush=True)
        z_text  = text_enc(batch['text'])    # (B, 768)
        z_audio = audio_enc(batch['audio'])  # (B, 1024)
        z_video = video_enc(batch['video'])  # (B, 768)
        text_emb_list.append(z_text)
        audio_emb_list.append(z_audio)
        video_emb_list.append(z_video)
        label_list.append(batch['label'].float())
    print()

    text_emb  = torch.cat(text_emb_list,  dim=0)   # (N, 768)
    audio_emb = torch.cat(audio_emb_list, dim=0)   # (N, 1024)
    video_emb = torch.cat(video_emb_list, dim=0)   # (N, 768)
    labels    = torch.cat(label_list,     dim=0)   # (N,) float32

    assert text_emb.shape[0] == N
    assert audio_emb.shape[0] == N
    assert video_emb.shape[0] == N

    return text_emb, audio_emb, video_emb, labels


def compute_norm_stats(text_emb, audio_emb, video_emb):
    """
    Per-modality z-score statistics computed on TRAINING embeddings only.
    Returns dict: {0: (mu, sigma), 1: (mu, sigma), 2: (mu, sigma)}
    All tensors on DEVICE.
    """
    stats = {}
    for m, Z in enumerate([text_emb, audio_emb, video_emb]):
        mu    = Z.mean(dim=0)
        sigma = Z.std(dim=0, unbiased=True)
        sigma = torch.where(sigma < 1e-8, torch.ones_like(sigma), sigma)
        stats[m] = (mu, sigma)
    return stats


def build_embedding(text_emb, audio_emb, video_emb,
                    variant: str, stats: dict = None) -> torch.Tensor:
    """
    Normalise and concatenate the three modality embeddings.

    variant = 'without'     : raw concatenation  →  (N, 768+1024+768)
    variant = 'just_scaling': divide each block by sqrt(d_m * M)
    variant = 'with'        : z-score then divide by sqrt(d_m * M)
                              → Euclidean distance ≡ modality-fair d(i,j)

    Returns (N, TEXT_DIM + AUDIO_DIM + VIDEO_DIM) float32 on DEVICE.
    """
    assert variant in ("without", "just_scaling", "with")
    M      = N_MODS
    parts  = []
    dims   = [TEXT_DIM, AUDIO_DIM, VIDEO_DIM]

    for m, (Z, dm) in enumerate(zip([text_emb, audio_emb, video_emb], dims)):
        Z = Z.float().clone()
        if variant == "with":
            mu, sigma = stats[m]
            Z = (Z - mu) / sigma
            Z = Z / (dm * M) ** 0.5
        elif variant == "just_scaling":
            Z = Z / (dm * M) ** 0.5
        parts.append(Z)

    return torch.cat(parts, dim=1)   # (N, 2560)


def _get_availability(dataset) -> np.ndarray:
    """
    Return boolean availability matrix (N, 3): [has_text, has_audio, has_video]
    from a MOSIDatasetRegression dataset, using its precomputed dict.
    """
    N    = len(dataset)
    avail = np.zeros((N, 3), dtype=bool)
    for idx in range(N):
        t, a, v = dataset.modality_availability[idx]
        avail[idx] = [t, a, v]
    return avail


def build_partial_embedding(raw_mods: list, present_mask: np.ndarray,
                             stats: dict, variant: str) -> np.ndarray:
    """
    Build a partial embedding vector using only present modalities.
    Missing modality blocks are zero-filled.
    The modality-fair scaling uses M' = number of present modalities so
    that the partial distance is well-defined.

    raw_mods    : list of 3 numpy arrays [(N,768),(N,1024),(N,768)]
                  (full arrays; rows are selected outside)
    present_mask: (3,) bool array
    stats       : normalisation stats from training set
    variant     : embedding variant string

    Returns (D,) float32 numpy vector.
    """
    dims   = [TEXT_DIM, AUDIO_DIM, VIDEO_DIM]
    M_pres = max(1, int(present_mask.sum()))
    parts  = []
    for m, (Z_row, dm) in enumerate(zip(raw_mods, dims)):
        Z = Z_row.copy().astype(np.float32)
        if not present_mask[m]:
            parts.append(np.zeros(dm, dtype=np.float32))
            continue
        if variant == "with":
            mu, sigma = stats[m]
            mu_np    = mu.cpu().numpy() if hasattr(mu, 'cpu') else mu
            sig_np   = sigma.cpu().numpy() if hasattr(sigma, 'cpu') else sigma
            Z = (Z - mu_np) / sig_np
            Z = Z / (dm * M_pres) ** 0.5
        elif variant == "just_scaling":
            Z = Z / (dm * M_pres) ** 0.5
        parts.append(Z)
    return np.concatenate(parts).astype(np.float32)


def get_all_embeddings(text_enc, audio_enc, video_enc,
                       train_ds, val_ds, test_ds):
    """
    Extract embeddings for all splits.  Everything stays in VRAM then is
    converted to numpy.

    Returns
    -------
    tr_emb, tr_labels, va_emb, va_labels, te_emb, te_labels  : full embeddings
    tr_raw, va_raw, te_raw   : list of 3 numpy arrays per split
                               [text_emb(N,768), audio_emb(N,1024), video_emb(N,768)]
    stats                    : normalisation stats dict (from training set)
    tr_avail, va_avail, te_avail : (N,3) bool arrays
    """
    print("\nExtracting embeddings …")

    print("  TRAIN …")
    tr_te, tr_ae, tr_ve, tr_labels_t = _encode_split(
        train_ds, text_enc, audio_enc, video_enc)

    print("  VAL   …")
    va_te, va_ae, va_ve, va_labels_t = _encode_split(
        val_ds, text_enc, audio_enc, video_enc)

    print("  TEST  …")
    te_te, te_ae, te_ve, te_labels_t = _encode_split(
        test_ds, text_enc, audio_enc, video_enc)

    # normalisation stats from training set only (on full present embeddings)
    stats = compute_norm_stats(tr_te, tr_ae, tr_ve)

    # raw per-modality numpy arrays
    tr_raw = [tr_te.cpu().float().numpy(),
              tr_ae.cpu().float().numpy(),
              tr_ve.cpu().float().numpy()]
    va_raw = [va_te.cpu().float().numpy(),
              va_ae.cpu().float().numpy(),
              va_ve.cpu().float().numpy()]
    te_raw = [te_te.cpu().float().numpy(),
              te_ae.cpu().float().numpy(),
              te_ve.cpu().float().numpy()]

    # modality availability masks
    tr_avail = _get_availability(train_ds)   # (N_tr, 3) bool
    va_avail = _get_availability(val_ds)
    te_avail = _get_availability(test_ds)

    def _build_partial_embs(raw_list, avail_mask):
        N   = raw_list[0].shape[0]
        out = np.zeros((N, sum([TEXT_DIM, AUDIO_DIM, VIDEO_DIM])),
                       dtype=np.float32)
        for i in range(N):
            out[i] = build_partial_embedding(
                [raw_list[m][i] for m in range(N_MODS)],
                avail_mask[i], stats, VARIANT)
        return out

    print("  Building partial embeddings (present mods only) …")
    tr_emb = _build_partial_embs(tr_raw, tr_avail)
    va_emb = _build_partial_embs(va_raw, va_avail)
    te_emb = _build_partial_embs(te_raw, te_avail)

    emb_dim = tr_emb.shape[1]
    print(f"  Embedding dim: {emb_dim}  (variant='{VARIANT}')\n")

    tr_labels = tr_labels_t.cpu().float().numpy()
    va_labels = va_labels_t.cpu().float().numpy()
    te_labels = te_labels_t.cpu().float().numpy()

    return (tr_emb, tr_labels, va_emb, va_labels, te_emb, te_labels,
            tr_raw, va_raw, te_raw, stats,
            tr_avail, va_avail, te_avail)


# =============================================================================
# RL PIPELINE
# =============================================================================

# ── kNN graph ─────────────────────────────────────────────────────────────────

def _build_knn_graph(emb: np.ndarray, k: int = K_GRAPH):
    """Directed k-NN adjacency list:  adj[i] = [(j, dist), …]"""
    N       = emb.shape[0]
    k_query = min(k + 1, N)
    index   = NearestNeighbors(n_neighbors=k_query, algorithm="auto",
                               metric="euclidean", n_jobs=-1)
    index.fit(emb)
    dists, idxs = index.kneighbors(emb)
    adj = [[] for _ in range(N)]
    for i in range(N):
        for pos in range(k_query):
            j = idxs[i, pos]
            d = dists[i, pos]
            if j != i and d > 1e-12:
                adj[i].append((j, float(d)))
    return adj


def _build_knn_index(tr_emb: np.ndarray, k_approx: int = K_APPROX):
    N_tr    = tr_emb.shape[0]
    k_query = min(k_approx + 1, N_tr)
    index   = NearestNeighbors(n_neighbors=k_query, algorithm="auto",
                               metric="euclidean", n_jobs=-1)
    index.fit(tr_emb)
    return index


# ── seed sets ─────────────────────────────────────────────────────────────────

def get_seeds_knn(knn_idxs, knn_dists, k: int = K_SEED_KNN):
    return [(int(knn_idxs[i]), float(knn_dists[i]))
            for i in range(min(k, len(knn_idxs)))]


def get_seeds_unn(q_vec, tr_emb, knn_idxs, knn_dists):
    nb_vecs     = tr_emb[knn_idxs]
    order       = np.argsort(knn_dists)
    useful_idx  = []
    useful_vecs = []
    for pos in order:
        d_ij   = knn_dists[pos]
        ni_vec = nb_vecs[pos]
        useless = any(np.linalg.norm(jv - ni_vec) < d_ij
                      for jv in useful_vecs)
        if not useless:
            useful_idx.append(pos)
            useful_vecs.append(ni_vec)
    return [(int(knn_idxs[p]), float(knn_dists[p])) for p in useful_idx]


# ── BFS candidate set ─────────────────────────────────────────────────────────

def build_candidate_set(seeds, target_label, tr_labels, adj,
                        max_depth: int = MAX_BFS_DEPTH, rng=None):
    """
    Dijkstra / BFS from seed nodes.  Regression version.

    Stops when BOTH conditions are met (max of B, C):
      B — current node depth >= MAX_BFS_DEPTH
      C — at least MAX_NODE2_RL nodes collected

    Oracle = the collected node whose label is closest to target_label
             in absolute value  (minimum |y_j - y_s|).

    Returns (shuffled_sequence, oracle_position) or (None, None).
    """
    visited = {}
    heap    = []
    for idx, dist in seeds:
        if idx not in visited:
            visited[idx] = (dist, 0)
            heapq.heappush(heap, (dist, idx, 0))

    sequence = []

    while heap:
        g_dist, node, depth = heapq.heappop(heap)
        if g_dist > visited.get(node, (float("inf"),))[0] + 1e-9:
            continue

        sequence.append((node, g_dist, depth))

        if depth < max_depth:
            for nb, ew in adj[node]:
                nd = g_dist + ew
                if nd < visited.get(nb, (float("inf"),))[0]:
                    visited[nb] = (nd, depth + 1)
                    heapq.heappush(heap, (nd, nb, depth + 1))

        cond_B = depth >= max_depth
        cond_C = len(sequence) >= MAX_NODE2_RL
        if cond_B and cond_C:
            break

    if not sequence:
        return None, None

    oracle_pos = min(
        range(len(sequence)),
        key=lambda p: abs(float(tr_labels[sequence[p][0]]) - float(target_label))
    )

    if rng is None:
        rng = np.random.RandomState()
    perm       = rng.permutation(len(sequence))
    sequence   = [sequence[p] for p in perm]
    oracle_pos = int(np.where(perm == oracle_pos)[0][0])
    return sequence, oracle_pos


# ── build all candidate sets ──────────────────────────────────────────────────

def build_all_sets(emb: np.ndarray, labels: np.ndarray, adj):
    N       = emb.shape[0]
    k_query = min(K_APPROX + 1, N)
    index   = NearestNeighbors(n_neighbors=k_query, algorithm="auto",
                               metric="euclidean", n_jobs=-1)
    index.fit(emb)
    knn_dists_all, knn_idxs_all = index.kneighbors(emb)

    all_sets    = []
    all_oracles = []
    set_sizes   = np.zeros(N, dtype=np.float32)
    rng         = np.random.RandomState(SEED)

    for i in range(N):
        mask  = knn_dists_all[i] > 1e-12
        idxs  = knn_idxs_all[i][mask]
        dists = knn_dists_all[i][mask]

        if STRATEGY == "knn":
            seeds = get_seeds_knn(idxs, dists)
        else:
            seeds = get_seeds_unn(emb[i], emb, idxs, dists)

        if not seeds:
            all_sets.append(None)
            all_oracles.append(None)
            continue

        cset, oracle = build_candidate_set(
            seeds, int(labels[i]), labels, adj, rng=rng)
        all_sets.append(cset)
        all_oracles.append(oracle)
        if cset is not None:
            set_sizes[i] = float(len(cset))

    multi = sum(1 for s in all_sets if s is not None and len(s) >= 2)
    valid_sizes = set_sizes[set_sizes > 0]
    print(f"  Sets built: {sum(1 for s in all_sets if s is not None)}/{N}"
          f"  |  ≥2 candidates: {multi}/{N}"
          f"  |  Mean size: {valid_sizes.mean():.1f}")
    return all_sets, all_oracles, set_sizes


# ── state / candidate features ────────────────────────────────────────────────

def build_features(q_vec, candidate_set, tr_emb, tr_labels):
    """
    State  (D+2): [f_s ; Var(y_B) ; |B|]
    Cand   (D+4): [f_v ; norm_dist ; norm_depth ; norm_rank ; lbl_dev]
    """
    D = q_vec.shape[0]
    n = len(candidate_set)

    cand_labels = np.array([float(tr_labels[idx])
                            for idx, _, _ in candidate_set], dtype=np.float32)
    mean_lbl  = cand_labels.mean()
    var_lbl   = cand_labels.var()
    max_depth = max(dep for _, _, dep in candidate_set) + 1e-9
    max_dist  = max(d   for _, d,   _ in candidate_set) + 1e-9

    state = np.concatenate(
        [q_vec, [var_lbl], [float(n)]]).astype(np.float32)

    sorted_by_dist = sorted(enumerate(candidate_set),
                            key=lambda x: x[1][1])
    rank_map = {orig: r for r, (orig, _) in enumerate(sorted_by_dist)}

    cand_feats = []
    for pos, (idx, dist, depth) in enumerate(candidate_set):
        feat = np.concatenate([
            tr_emb[idx],
            [dist  / max_dist],
            [depth / max_depth],
            [rank_map[pos] / max(n - 1, 1)],
            [float(tr_labels[idx]) - mean_lbl],
        ]).astype(np.float32)
        cand_feats.append(feat)

    return state, np.stack(cand_feats)


# ── PPO scoring head ──────────────────────────────────────────────────────────

class ScoringMLP(nn.Module):
    """score_i = MLP_θ([h_s ; h_i ; h_s ⊙ h_i])"""
    def __init__(self, state_dim: int, cand_dim: int, hidden: int = 256):
        super().__init__()
        self.query_enc = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.cand_enc = nn.Sequential(
            nn.Linear(cand_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.score_head = nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state, cand_feats):
        squeeze = state.dim() == 1
        if squeeze:
            state      = state.unsqueeze(0)
            cand_feats = cand_feats.unsqueeze(0)
        B, K, _ = cand_feats.shape
        h_o     = self.query_enc(state)
        h_o_exp = h_o.unsqueeze(1).expand(-1, K, -1)
        h_i     = self.cand_enc(cand_feats)
        fused   = torch.cat([h_o_exp, h_i, h_o_exp * h_i], dim=-1)
        scores  = self.score_head(fused).squeeze(-1)
        return scores.squeeze(0) if squeeze else scores


# ── PPO update ────────────────────────────────────────────────────────────────

def ppo_update(policy, optimizer, rollout,
               clip: float = PPO_CLIP, ent_coef: float = ENT_COEF):
    """
    Single minibatch PPO update.

    old_log_pis are collected ONCE at the start of the epoch and kept
    frozen throughout all minibatch updates in that epoch.
    """
    states, cands_list, actions, old_log_pis, advantages = rollout
    losses = []
    for i in range(len(states)):
        scores  = policy(states[i], cands_list[i])
        log_pi  = F.log_softmax(scores, dim=-1)
        pi      = log_pi.exp()
        new_lp  = log_pi[actions[i]]
        ratio   = torch.exp(new_lp - old_log_pis[i].detach())
        adv     = advantages[i].detach()
        surr1   = ratio * adv
        surr2   = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
        entropy = -(pi * log_pi).sum()
        losses.append(-torch.min(surr1, surr2) - ent_coef * entropy)
    total_loss = torch.stack(losses).mean()
    optimizer.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
    return total_loss.item()


# ── inference ─────────────────────────────────────────────────────────────────

def _infer_one(q_vec, knn_idxs, knn_dists, tr_emb, tr_labels,
               adj, tr_set_sizes, policy):
    mask  = knn_dists > 1e-12
    idxs  = knn_idxs[mask]
    dists = knn_dists[mask]
    if len(idxs) == 0:
        return float(tr_labels[0])

    if STRATEGY == "knn":
        seeds = get_seeds_knn(idxs, dists)
    else:
        seeds = get_seeds_unn(q_vec, tr_emb, idxs, dists)

    if not seeds:
        return float(tr_labels[idxs[0]])

    target_sz = max(1, int(max(float(tr_set_sizes[idx]) for idx, _ in seeds)))

    visited = {}
    heap    = []
    for idx, dist in seeds:
        if idx not in visited:
            visited[idx] = (dist, 0)
            heapq.heappush(heap, (dist, idx, 0))

    sequence = []
    while heap and len(sequence) < target_sz:
        g_dist, node, depth = heapq.heappop(heap)
        if g_dist > visited.get(node, (float("inf"),))[0] + 1e-9:
            continue
        sequence.append((node, g_dist, depth))
        for nb, ew in adj[node]:
            nd = g_dist + ew
            if nd < visited.get(nb, (float("inf"),))[0]:
                visited[nb] = (nd, depth + 1)
                heapq.heappush(heap, (nd, nb, depth + 1))

    if not sequence:
        return float(tr_labels[idxs[0]])
    if len(sequence) == 1:
        return float(tr_labels[sequence[0][0]])

    state, cand_feats = build_features(q_vec, sequence, tr_emb, tr_labels)
    sc   = policy(torch.tensor(state, device=DEVICE),
                  torch.tensor(cand_feats, device=DEVICE))
    best = int(sc.argmax().item())
    return float(tr_labels[sequence[best][0]])


def evaluate_policy(policy, va_emb, va_labels, tr_emb, tr_labels,
                    adj, tr_set_sizes):
    """Returns val MSE (lower is better)."""
    policy.eval()
    index = _build_knn_index(tr_emb)
    knn_dists_all, knn_idxs_all = index.kneighbors(va_emb)
    preds = []
    with torch.no_grad():
        for qi in range(len(va_labels)):
            preds.append(_infer_one(
                va_emb[qi], knn_idxs_all[qi], knn_dists_all[qi],
                tr_emb, tr_labels, adj, tr_set_sizes, policy))
    preds = np.array(preds, dtype=np.float32)
    mse   = float(np.mean((preds - va_labels) ** 2))
    return mse


def predict_test(policy, te_emb, te_labels, tr_emb, tr_labels,
                 adj, tr_set_sizes):
    index = _build_knn_index(tr_emb)
    knn_dists_all, knn_idxs_all = index.kneighbors(te_emb)
    policy.eval()
    preds = []
    with torch.no_grad():
        for qi in range(len(te_labels)):
            preds.append(_infer_one(
                te_emb[qi], knn_idxs_all[qi], knn_dists_all[qi],
                tr_emb, tr_labels, adj, tr_set_sizes, policy))
    return np.array(preds, dtype=np.float32)


# ── PPO training loop ─────────────────────────────────────────────────────────

def train_policy(tr_emb, tr_labels, tr_sets, tr_oracles,
                 va_emb, va_labels, adj, tr_set_sizes,
                 te_emb=None, te_labels=None):

    D         = tr_emb.shape[1]
    state_dim = D + 2
    cand_dim  = D + 4

    policy    = ScoringMLP(state_dim, cand_dim).to(DEVICE)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

    best_val_mse = float("inf")
    best_state   = None

    valid_tr = [i for i in range(len(tr_emb))
                if tr_sets[i] is not None
                and len(tr_sets[i]) >= 2
                and tr_oracles[i] is not None]

    print(f"  Valid training points (≥2 candidates): {len(valid_tr)}")
    print(f"\nTraining PPO ({PPO_EPOCHS} epochs) …")

    for epoch in range(PPO_EPOCHS):
        epoch_rng = np.random.RandomState(SEED + epoch)

        epoch_sets    = {}
        epoch_oracles = {}
        for i in valid_tr:
            cset = tr_sets[i]
            perm = epoch_rng.permutation(len(cset))
            new_cset           = [cset[p] for p in perm]
            orig_oracle_node   = cset[tr_oracles[i]][0]
            new_oracle         = int(np.where(
                np.array([n for n, _, _ in new_cset]) == orig_oracle_node
            )[0][0])
            epoch_sets[i]    = new_cset
            epoch_oracles[i] = new_oracle

        epoch_rng.shuffle(valid_tr)

        # ── Step 1: collect full epoch rollout with policy FROZEN ─────────
        states, cands_l, actions, old_lps, advantages = [], [], [], [], []

        policy.eval()
        with torch.no_grad():
            for i in valid_tr:
                cset   = epoch_sets[i]
                oracle = epoch_oracles[i]
                state, cand_feats = build_features(
                    tr_emb[i], cset, tr_emb, tr_labels)
                s_t  = torch.tensor(state,      device=DEVICE)
                cf_t = torch.tensor(cand_feats, device=DEVICE)
                scores = policy(s_t, cf_t)
                log_pi = F.log_softmax(scores, dim=-1)
                probs  = log_pi.exp()
                action = int(torch.multinomial(probs, 1).item())
                old_lp = log_pi[action]
                y_s    = float(tr_labels[i])
                K = len(cset)
                if K <= 1:
                    reward = 0.0
                else:
                    errors      = [abs(float(tr_labels[cset[j][0]]) - y_s)
                                   for j in range(K)]
                    sorted_idx  = np.argsort(errors)   # best → worst
                    rank_map_r  = {int(sorted_idx[r]): r for r in range(K)}
                    rank_action = rank_map_r[action]
                    rank_oracle = rank_map_r[oracle]
                    reward      = float(rank_oracle - rank_action) / (K - 1)
                states.append(s_t)
                cands_l.append(cf_t)
                actions.append(action)
                old_lps.append(old_lp.detach())
                advantages.append(
                    torch.tensor(reward, dtype=torch.float32, device=DEVICE))

        # ── Step 2: PPO minibatch updates with frozen old_log_pis ─────────
        policy.train()
        M    = len(states)
        perm = epoch_rng.permutation(M)
        ep_loss, nb = 0.0, 0
        for start in range(0, M, MINIBATCH_SIZE):
            mb = perm[start: start + MINIBATCH_SIZE]
            rollout = (
                [states[j]  for j in mb], [cands_l[j] for j in mb],
                [actions[j] for j in mb], [old_lps[j] for j in mb],
                [advantages[j] for j in mb],
            )
            ep_loss += ppo_update(policy, optimizer, rollout)
            nb += 1

        val_mse = evaluate_policy(policy, va_emb, va_labels,
                                  tr_emb, tr_labels, adj, tr_set_sizes)
        marker = " ← best" if val_mse < best_val_mse else ""

        debug_str = ""
        if te_emb is not None and te_labels is not None:
            te_preds  = predict_test(policy, te_emb, te_labels,
                                     tr_emb, tr_labels, adj, tr_set_sizes)
            te_mse    = float(np.mean((te_preds - te_labels) ** 2))
            debug_str = f"  [DEBUG test_mse={te_mse:.4f}]"

        print(f"  Epoch {epoch+1:>3d}/{PPO_EPOCHS}  "
              f"loss={ep_loss/max(nb,1):.4f}  "
              f"val_mse={val_mse:.4f}{marker}{debug_str}")

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state   = copy.deepcopy(policy.state_dict())
            torch.save(best_state, RL_CKPT)

    print(f"\n  Best val MSE: {best_val_mse:.4f}  →  {RL_CKPT}")
    policy.load_state_dict(best_state)
    return policy


# =============================================================================
# FUSION HEAD  —  attends over top-K' RL-ranked candidates
# =============================================================================

class FusionHead(nn.Module):
    """
    Small cross-attention module that produces a weighted-average label
    prediction from the top-K' candidates selected by the RL policy.

    Architecture
    ------------
    Query  : linear projection of the query embedding  (D → fusion_hidden)
    Keys   : linear projection of each candidate emb   (D → fusion_hidden)
    Values : candidate labels                           (1-dim, kept scalar)

    Attention scores → softmax weights → weighted sum of candidate labels.
    A small 2-layer MLP on top of the context vector refines the prediction.
    """
    def __init__(self, emb_dim: int,
                 hidden:  int = FUSION_HIDDEN,
                 n_heads: int = FUSION_HEADS,
                 topk:    int = FUSION_TOPK):
        super().__init__()
        self.topk    = topk
        self.n_heads = n_heads
        self.d_head  = hidden // n_heads
        assert hidden % n_heads == 0, "FUSION_HIDDEN must be divisible by FUSION_HEADS"

        self.q_proj  = nn.Linear(emb_dim, hidden, bias=False)
        self.k_proj  = nn.Linear(emb_dim, hidden, bias=False)
        self.scale   = self.d_head ** -0.5

        self.context_proj = nn.Linear(hidden, hidden)
        self.refine = nn.Sequential(
            nn.Linear(hidden + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, q_emb: torch.Tensor,
                cand_embs: torch.Tensor,
                cand_labels: torch.Tensor) -> torch.Tensor:
        """
        q_emb       : (D,)          query embedding
        cand_embs   : (K', D)       top-K' candidate embeddings
        cand_labels : (K',)         their labels (float)

        Returns scalar prediction.
        """
        K   = cand_embs.shape[0]
        H   = self.n_heads
        Dh  = self.d_head

        q = self.q_proj(q_emb)           # (hidden,)
        k = self.k_proj(cand_embs)       # (K', hidden)

        q = q.view(H, Dh)                # (H, Dh)
        k = k.view(K, H, Dh)            # (K', H, Dh)

        scores = torch.einsum("hd,khd->hk", q, k) * self.scale
        weights = F.softmax(scores, dim=-1)          # (H, K')
        avg_w   = weights.mean(dim=0)                # (K',)

        attended_label = (avg_w * cand_labels).sum().unsqueeze(0)  # (1,)

        k_flat  = k.reshape(K, H * Dh)                              # (K', hidden)
        context = (avg_w.unsqueeze(-1) * k_flat).sum(dim=0)         # (hidden,)
        context = F.relu(self.context_proj(context))                 # (hidden,)

        refined = self.refine(
            torch.cat([context, attended_label], dim=-1))            # (1,)
        return refined.squeeze(-1)                                   # scalar


@torch.no_grad()
def _get_topk_candidates(q_vec, sequence, tr_emb, tr_labels, policy, topk):
    """
    Run the RL policy over `sequence`, return the top-`topk` candidates
    sorted by policy score (highest score first).

    Returns
    -------
    cand_embs   : (min(topk, K), D) float32 numpy
    cand_labels : (min(topk, K),)   float32 numpy
    """
    if len(sequence) == 0:
        return None, None

    state, cand_feats = build_features(q_vec, sequence, tr_emb, tr_labels)
    scores = policy(
        torch.tensor(state,      device=DEVICE),
        torch.tensor(cand_feats, device=DEVICE),
    )
    top_indices = scores.argsort(descending=True)[:topk].cpu().numpy()

    cand_embs   = np.stack([tr_emb[sequence[j][0]]   for j in top_indices])
    cand_labels = np.array([float(tr_labels[sequence[j][0]]) for j in top_indices],
                           dtype=np.float32)
    return cand_embs, cand_labels


def train_fusion(policy, tr_emb, tr_labels, tr_sets,
                 va_emb, va_labels, adj, tr_set_sizes,
                 te_emb=None, te_labels=None):
    """
    Train the FusionHead on the training set.

    For each training point:
      1. Build the BFS candidate set (precomputed in tr_sets).
      2. Ask the frozen RL policy to rank them.
      3. Take the top-FUSION_TOPK candidates.
      4. Supervised MSE loss: FusionHead(q, top-K') vs y_s.

    Checkpoint by lowest val MSE.
    """
    print("\n" + "=" * 60)
    print("FUSION HEAD training")
    print("=" * 60)

    D     = tr_emb.shape[1]
    head  = FusionHead(emb_dim=D).to(DEVICE)
    optim = torch.optim.Adam(head.parameters(), lr=FUSION_LR)
    policy.eval()

    valid = [i for i in range(len(tr_emb))
             if tr_sets[i] is not None and len(tr_sets[i]) >= 1]

    best_val_mse  = float("inf")
    best_state    = None

    for epoch in range(FUSION_EPOCHS):
        head.train()
        rng  = np.random.RandomState(SEED + 1000 + epoch)
        perm = rng.permutation(len(valid))
        ep_loss, nb = 0.0, 0

        for start in range(0, len(valid), FUSION_BATCH):
            mb_idx = [valid[perm[j]]
                      for j in range(start, min(start + FUSION_BATCH, len(valid)))]
            batch_loss = []

            for i in mb_idx:
                cset = tr_sets[i]
                ce, cl = _get_topk_candidates(
                    tr_emb[i], cset, tr_emb, tr_labels, policy, FUSION_TOPK)
                if ce is None or len(ce) == 0:
                    continue
                q_t  = torch.tensor(tr_emb[i], dtype=torch.float32, device=DEVICE)
                ce_t = torch.tensor(ce,         dtype=torch.float32, device=DEVICE)
                cl_t = torch.tensor(cl,         dtype=torch.float32, device=DEVICE)
                pred = head(q_t, ce_t, cl_t)
                target = torch.tensor(float(tr_labels[i]),
                                      dtype=torch.float32, device=DEVICE)
                batch_loss.append(F.mse_loss(pred, target))

            if not batch_loss:
                continue
            loss = torch.stack(batch_loss).mean()
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optim.step()
            ep_loss += loss.item()
            nb += 1

        val_mse = _evaluate_fusion(head, policy, va_emb, va_labels,
                                   tr_emb, tr_labels, adj, tr_set_sizes)
        marker = " ← best" if val_mse < best_val_mse else ""
        debug_str = ""
        if te_emb is not None and te_labels is not None:
            te_preds = predict_test_fusion(head, policy, te_emb, te_labels,
                                           tr_emb, tr_labels, adj, tr_set_sizes)
            te_mse   = float(np.mean((te_preds - te_labels) ** 2))
            debug_str = f"  [DEBUG test_mse={te_mse:.4f}]"
        print(f"  Epoch {epoch+1:>3d}/{FUSION_EPOCHS}  "
              f"loss={ep_loss/max(nb,1):.4f}  "
              f"val_mse={val_mse:.4f}{marker}{debug_str}")

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state   = copy.deepcopy(head.state_dict())
            torch.save(best_state, FUSION_CKPT)

    print(f"\n  Best fusion val MSE: {best_val_mse:.4f}  →  {FUSION_CKPT}")
    head.load_state_dict(best_state)
    return head


def _infer_one_fusion(q_vec, knn_idxs, knn_dists, tr_emb, tr_labels,
                      adj, tr_set_sizes, policy, head):
    """
    Inference with fusion head.
    Builds BFS candidate set, ranks with policy, feeds top-K' to FusionHead.
    """
    mask  = knn_dists > 1e-12
    idxs  = knn_idxs[mask]
    dists = knn_dists[mask]
    if len(idxs) == 0:
        return float(tr_labels[0])

    seeds = get_seeds_unn(q_vec, tr_emb, idxs, dists) if STRATEGY == "unn" \
            else get_seeds_knn(idxs, dists)
    if not seeds:
        return float(tr_labels[idxs[0]])

    target_sz = max(1, int(max(float(tr_set_sizes[idx]) for idx, _ in seeds)))
    visited, heap = {}, []
    for idx, dist in seeds:
        if idx not in visited:
            visited[idx] = (dist, 0)
            heapq.heappush(heap, (dist, idx, 0))

    sequence = []
    while heap and len(sequence) < target_sz:
        g_dist, node, depth = heapq.heappop(heap)
        if g_dist > visited.get(node, (float("inf"),))[0] + 1e-9:
            continue
        sequence.append((node, g_dist, depth))
        for nb, ew in adj[node]:
            nd = g_dist + ew
            if nd < visited.get(nb, (float("inf"),))[0]:
                visited[nb] = (nd, depth + 1)
                heapq.heappush(heap, (nd, nb, depth + 1))

    if not sequence:
        return float(tr_labels[idxs[0]])

    ce, cl = _get_topk_candidates(q_vec, sequence, tr_emb, tr_labels,
                                  policy, FUSION_TOPK)
    if ce is None or len(ce) == 0:
        return float(tr_labels[sequence[0][0]])

    q_t  = torch.tensor(q_vec, dtype=torch.float32, device=DEVICE)
    ce_t = torch.tensor(ce,    dtype=torch.float32, device=DEVICE)
    cl_t = torch.tensor(cl,    dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        pred = head(q_t, ce_t, cl_t)
    return float(pred.item())


def _evaluate_fusion(head, policy, emb, labels, tr_emb, tr_labels,
                     adj, tr_set_sizes):
    head.eval()
    index = _build_knn_index(tr_emb)
    knn_dists_all, knn_idxs_all = index.kneighbors(emb)
    preds = []
    with torch.no_grad():
        for qi in range(len(labels)):
            preds.append(_infer_one_fusion(
                emb[qi], knn_idxs_all[qi], knn_dists_all[qi],
                tr_emb, tr_labels, adj, tr_set_sizes, policy, head))
    return float(np.mean((np.array(preds, dtype=np.float32) - labels) ** 2))


def predict_test_fusion(head, policy, te_emb, te_labels,
                        tr_emb, tr_labels, adj, tr_set_sizes):
    index = _build_knn_index(tr_emb)
    knn_dists_all, knn_idxs_all = index.kneighbors(te_emb)
    head.eval()
    preds = []
    with torch.no_grad():
        for qi in range(len(te_labels)):
            preds.append(_infer_one_fusion(
                te_emb[qi], knn_idxs_all[qi], knn_dists_all[qi],
                tr_emb, tr_labels, adj, tr_set_sizes, policy, head))
    return np.array(preds, dtype=np.float32)


# =============================================================================
# LABEL CONVERSION HELPERS  (for final test metrics)
# =============================================================================

def _to_binary(scores: np.ndarray) -> np.ndarray:
    """Continuous scores → {0, 1}  (1 = positive sentiment >= 0)."""
    return (scores >= BINARISE_THRESHOLD).astype(int)


def _to_7class(scores: np.ndarray) -> np.ndarray:
    """
    Continuous scores → 7-class labels {0,…,6} by binning into
    [-3,-2), [-2,-1), [-1,0), [0,1), [1,2), [2,3), [3,+inf).
    """
    bins = np.array(SEVEN_CLASS_BINS)
    return np.digitize(scores, bins[1:], right=False).astype(int)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"Device   : {DEVICE}")
    print(f"Variant  : {VARIANT}")
    print(f"Strategy : {STRATEGY}")
    print(f"Config   : {MISSING_CONFIG}\n")

    # ── load datasets ─────────────────────────────────────────────────────────
    print("Loading MOSI datasets …")
    common = dict(
        audio_dir      = AUDIO_DIR,
        video_dir      = VIDEO_DIR,
        text_dir       = TEXT_DIR,
        split_file     = SPLIT_FILE,
        missing_config = MISSING_CONFIG,
        seed           = SEED,
    )
    train_ds = MOSIDatasetRegression(**common, split='train')
    val_ds   = MOSIDatasetRegression(**common, split='val')
    test_ds  = MOSIDatasetRegression(**common, split='test')
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}\n")

    # ── load frozen encoders onto DEVICE ──────────────────────────────────────
    print("Loading frozen encoders onto GPU …")
    text_enc  = FrozenTextEncoder().to(DEVICE)
    audio_enc = FrozenAudioEncoder().to(DEVICE)
    video_enc = FrozenVideoEncoder(num_frames=NUM_FRAMES).to(DEVICE)
    print("  Done.\n")

    # ── extract all embeddings ────────────────────────────────────────────────
    (tr_emb, tr_labels, va_emb, va_labels, te_emb, te_labels,
     tr_raw, va_raw, te_raw, stats,
     tr_avail, va_avail, te_avail) = get_all_embeddings(
        text_enc, audio_enc, video_enc,
        train_ds, val_ds, test_ds)

    del text_enc, audio_enc, video_enc
    torch.cuda.empty_cache()
    print("  Encoders freed from VRAM.\n")

    # ── build training k-NN graph ──────────────────────────────────────────────
    print(f"Building training k-NN graph (k={K_GRAPH}) …")
    adj = _build_knn_graph(tr_emb, k=K_GRAPH)
    print("  Done.\n")

    # ── build BFS candidate sets ───────────────────────────────────────────────
    print(f"Building BFS candidate sets (strategy='{STRATEGY}') …")
    all_sets, all_oracles, set_sizes = build_all_sets(
        tr_emb, tr_labels, adj)
    print()

    # ── train RL policy ────────────────────────────────────────────────────────
    print("=" * 60)
    print("PHASE — RL neighbour selection (PPO)")
    print("=" * 60)
    policy = train_policy(
        tr_emb, tr_labels, all_sets, all_oracles,
        va_emb, va_labels, adj, set_sizes,
        te_emb=te_emb, te_labels=te_labels,
    )

    # ── RL-only final test ────────────────────────────────────────────────────
    print("\nRunning final test inference (RL-only) …")
    preds_rl = predict_test(policy, te_emb, te_labels,
                            tr_emb, tr_labels, adj, set_sizes)

    # ── train fusion head ─────────────────────────────────────────────────────
    head = train_fusion(policy, tr_emb, tr_labels, all_sets,
                        va_emb, va_labels, adj, set_sizes,
                        te_emb=te_emb, te_labels=te_labels)

    print("\nRunning final test inference (RL + Fusion) …")
    preds_fus = predict_test_fusion(head, policy, te_emb, te_labels,
                                    tr_emb, tr_labels, adj, set_sizes)

    def _report(label, preds):
        test_mse     = float(np.mean((preds - te_labels) ** 2))
        preds_bin    = _to_binary(preds)
        labels_bin   = _to_binary(te_labels)
        f1_bin_micro = f1_score(labels_bin, preds_bin, average="micro")
        f1_bin_macro = f1_score(labels_bin, preds_bin, average="macro")
        preds_7      = _to_7class(preds)
        labels_7     = _to_7class(te_labels)
        f1_7_micro   = f1_score(labels_7, preds_7, average="micro", zero_division=0)
        f1_7_macro   = f1_score(labels_7, preds_7, average="macro", zero_division=0)
        sep = "─" * 56
        print(f"\n{sep}")
        print(f"  {label}")
        print(f"  variant='{VARIANT}'  strategy='{STRATEGY}'")
        print(f"  MSE                  : {test_mse:.4f}")
        print(f"  F1-micro  (binary)   : {f1_bin_micro:.4f}")
        print(f"  F1-macro  (binary)   : {f1_bin_macro:.4f}")
        print(f"  F1-micro  (7-class)  : {f1_7_micro:.4f}")
        print(f"  F1-macro  (7-class)  : {f1_7_macro:.4f}")
        print(f"{sep}\n")

    _report(f"RL-BFS  (hard selection) {MISSING_CONFIG}",        preds_rl)
    _report(f"RL-BFS + Fusion Head (attended) {MISSING_CONFIG}",  preds_fus)


if __name__ == "__main__":
    main()