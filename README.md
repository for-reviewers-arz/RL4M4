# RL4M⁴ — Reinforcement Learning for Multi-Modal Missing-Modality Retrieval

Relational, retrieval-based prediction under **arbitrary missing modalities**, using a
**modality-fair distance**, a **BFS-augmented candidate graph**, an **RL (PPO) neighbour-selection
policy**, and a lightweight **cross-attention fusion head**.

This repository contains a reference implementation of the method for **CMU-MOSI**
(continuous sentiment regression, 3 modalities: text / audio / video), with the same
pipeline described in the paper also evaluated on MM-IMDb (image + text, multi-label
classification) and a 5-modal Sleep-EDF polysomnography dataset.

> **Status:** research code accompanying an internal method write-up. Paths, checkpoints,
> and dataset locations are placeholders — see [Configuration](#configuration) before running.

---

## Table of Contents

- [Method Overview](#method-overview)
- [Repository Structure](#repository-structure)
- [Datasets](#datasets)
- [Missing-Modality Simulation](#missing-modality-simulation)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Pipeline Details](#pipeline-details)
- [Results](#results)
- [Known Deviations from the Method Write-Up](#known-deviations-from-the-method-write-up)
- [Citation](#citation)

---

## Method Overview

Given frozen, modality-specific pretrained encoders, each sample is embedded per modality
and the embeddings are z-score normalised using training-set statistics only. A
**modality-fair distance** then averages each modality's per-dimension squared error, so
that no modality dominates just because its encoder happens to output a larger vector.

From this distance:

1. A **k-NN seed set** is retrieved for each query (either plain k-NN or a redundancy-pruned
   variant, "UNN").
2. The seed set is **expanded via a depth-bounded Dijkstra/BFS search** over a training
   k-NN graph, producing a richer, label-diverse candidate pool.
3. An **RL policy** (PPO, attention-style scoring head) learns to pick the single most
   informative candidate from that pool, trained against an oracle (the closest-matching
   label reachable in the candidate graph).
4. A **cross-attention fusion head** then softens this hard selection: it attends over the
   top-K′ RL-ranked candidates and predicts a refined label instead of committing to one
   neighbour.

Missing modalities are handled by **zero-imputation** — no architecture changes,
gating networks, or generative modality hallucination are required.

## Repository Structure

```
.
├── mosi_reg.py     # CMU-MOSI regression Dataset with configurable missing-modality simulation
├── rl.py           # Full pipeline: encoders → embeddings → distance → BFS graph →
│                   #   PPO policy training → fusion head training → evaluation
└── README.md
```

## Datasets

| Dataset | Modalities | Task | Notes |
|---|---|---|---|
| **CMU-MOSI** | Text, Audio, Video | Regression (sentiment ∈ [-3, 3]) | Reference implementation in this repo |
| MM-IMDb | Image, Text | Multi-label classification (27 genres) | Same pipeline, classification head |
| Sleep-EDF | 5 physiological signals | 5-class sleep-stage classification | Appendix results, `pm`/`ps` missingness control |

For CMU-MOSI, `mosi_reg.py` expects:
- Segmented 16kHz WAV audio
- Segmented video clips
- `.annotprocessed` transcript files
- A JSON split file with `train` / `val` / `test` lists, each item having a `name` and a
  continuous `label`

Update the dataset paths in `rl.py`'s `PARAMETERS` block (`AUDIO_DIR`, `VIDEO_DIR`,
`TEXT_DIR`, `SPLIT_FILE`) before running.

## Missing-Modality Simulation

`mosi_reg.py` deterministically (seeded) assigns per-sample modality availability
according to a `missing_config` string, in one of two formats:

**Simple** — `T_text_A_audio_V_video`, independent per-modality availability:
```
"20_text_100_audio_100_video"   # text present 20% of the time, audio & video always present
"100_text_100_audio_100_video"  # fully complete dataset (no missingness)
```

**Complex** — `complex_TAV_T_A_V_TA_TV_AV`, an exhaustive joint distribution over the 7
non-empty modality-presence patterns (must sum to 100):
```
"complex_20_20_20_10_10_10_10"
#   20% all three modalities present
#   20% text only
#   20% audio only
#   10% video only
#   10% text + audio
#   10% text + video
#   10% audio + video
```

Since CMU-MOSI only ships one fully multimodal split, missingness is *simulated* by
zeroing out the corresponding raw inputs before encoding — the dataset itself is never
duplicated or altered on disk.

## Installation

```bash
git clone <this-repo>
cd <this-repo>
pip install torch torchaudio torchvision transformers scikit-learn opencv-python pillow numpy
```

Requires a CUDA-capable GPU for practical runtimes (BERT / WavLM-Large / CLIP encoding of
the full training set, kept resident in VRAM during Phase 2).

## Configuration

All tuneable parameters live in the `PARAMETERS` block at the top of `rl.py`:

| Group | Key parameters |
|---|---|
| Dataset paths | `AUDIO_DIR`, `VIDEO_DIR`, `TEXT_DIR`, `SPLIT_FILE` |
| Missingness | `MISSING_CONFIG`, `SEED` |
| Embedding variant | `VARIANT`: `"without"` (raw concat), `"just_scaling"`, `"with"` (z-score + modality-fair scaling — **this is the distance described in the method**) |
| Graph / BFS | `STRATEGY` (`knn`/`unn`), `K_APPROX`, `K_SEED_KNN`, `K_GRAPH`, `MAX_BFS_DEPTH`, `MAX_NODE2_RL` |
| PPO | `PPO_EPOCHS`, `PPO_CLIP`, `LR`, `ENT_COEF`, `MINIBATCH_SIZE` |
| Fusion head | `FUSION_TOPK`, `FUSION_HIDDEN`, `FUSION_HEADS`, `FUSION_EPOCHS`, `FUSION_LR`, `FUSION_BATCH` |

> ⚠️ **`VARIANT` defaults to `"without"`**, i.e. raw concatenated embeddings with **no**
> modality-fair normalisation. Set `VARIANT = "with"` to reproduce the distance metric
> described in the method.

## Usage

```bash
python rl.py
```

This runs the full pipeline end-to-end:
1. Load train/val/test splits and simulate missing modalities.
2. Extract frozen BERT / WavLM-Large / CLIP embeddings (single pass, GPU-resident).
3. Build the training k-NN graph and BFS-augmented candidate sets.
4. Train the PPO neighbour-selection policy, checkpointing on best validation MSE.
5. Train the cross-attention fusion head on top of the frozen, trained policy.
6. Report test-set **MSE**, **binary F1** (micro/macro), and **7-class F1** (micro/macro)
   for both the RL-only (hard selection) and RL+Fusion predictions.

## Pipeline Details

| Component | Implementation |
|---|---|
| Encoders | Frozen `bert-base-uncased` (768-d, CLS token), `microsoft/wavlm-large` (1024-d, mean-pooled), `openai/clip-vit-base-patch32` (768-d, mean-pooled over 8 sampled frames) |
| Normalisation | Per-modality z-score fit on training embeddings only; std clamped at `1e-8` |
| Distance | Modality-fair distance realised as plain Euclidean distance over the concatenated, per-block-scaled embedding (`z̃ / sqrt(d_m · M)`) |
| Candidate graph | Directed k-NN adjacency (`K_GRAPH`), Dijkstra expansion up to `MAX_BFS_DEPTH` hops or `MAX_NODE2_RL` nodes |
| Policy | Two-branch MLP encoder (query / candidate) → `[h_s; h_v; h_s⊙h_v]` → scoring MLP → softmax over candidates |
| RL training | PPO with clipped surrogate objective, entropy bonus, no value baseline (`Â_t = r(s,a_t)`) |
| Reward | Rank-based: normalised difference between the oracle's and the chosen candidate's rank-by-label-error |
| Fusion head | Linear query/key projections, multi-head scaled dot-product attention over top-`K'` RL-ranked candidates, attended label + attention context fed to a 2-layer MLP |

## Results

Selected results from the accompanying method write-up (see paper for full tables and
ablations across all missingness configurations and datasets):

**CMU-MOSI — F1-micro (binary sentiment), by modality-missingness config**

| Method | 100T-100A-100V | 100T-100A-20V | 100T-20A-100V | 100T-20A-20V | 20T-100A-100V | 20T-100A-20V | 20T-20A-100V | complex-20-20-20-10-10-10-10 |
|---|---|---|---|---|---|---|---|---|
| ShaSpec | 0.35 | 0.33 | 0.32 | 0.30 | 0.28 | 0.25 | 0.27 | 0.29 |
| M³Care | 0.36 | 0.34 | 0.33 | 0.31 | 0.29 | 0.26 | 0.28 | 0.30 |
| ICL-CA | 0.71 | 0.56 | 0.78 | 0.56 | 0.67 | 0.56 | 0.65 | 0.68 |
| SimMLM | 0.77 | 0.76 | 0.77 | 0.45 | 0.69 | 0.54 | 0.63 | 0.68 |
| **RL4M⁴ (Ours)** | **0.82** | **0.78** | **0.81** | **0.78** | **0.66** | **0.60** | **0.69** | **0.69** |

**5-Modal Sleep-EDF — Accuracy**, across sample-missing fraction `pm` and per-sample
missing-modality ratio `ps`:

| Method | pm=0.4, ps=0.2 | 0.4 | 0.6 | 0.8 | pm=0.6, ps=0.2 | 0.4 | 0.6 | 0.8 |
|---|---|---|---|---|---|---|---|---|
| ShaSpec | 0.32 | 0.34 | 0.34 | 0.34 | 0.33 | 0.33 | 0.32 | 0.32 |
| M³Care | 0.39 | 0.39 | 0.38 | 0.37 | 0.38 | 0.37 | 0.36 | 0.36 |
| ICL-CA | 0.39 | 0.36 | 0.36 | 0.35 | 0.36 | 0.36 | 0.35 | 0.35 |
| SimMLM | 0.39 | 0.38 | 0.38 | 0.38 | 0.37 | 0.36 | 0.35 | 0.34 |
| **RL4M⁴ (Ours)** | **0.45** | **0.45** | **0.45** | **0.45** | **0.45** | **0.45** | **0.45** | **0.45** |

RL4M⁴'s accuracy stays essentially flat across all eight configurations, consistent with
the claim that the relational, graph-based selection mechanism is largely insensitive to
how many of the modalities are dropped per sample.

## Known Deviations from the Method Write-Up

For transparency, a few places where this implementation adapts or departs from the
literal equations in the method description:

- **Reward function.** The write-up defines the PPO reward as the task-loss difference
  between the chosen candidate and the oracle, `r = -L(y_a, y_s) + L(y_oracle, y_s)`. For
  the regression setting, `rl.py` instead uses a **rank-based** reward — the normalised
  difference between the oracle's and the chosen candidate's rank by label-error — which
  keeps rewards bounded in `[-1, 1]` regardless of the label scale.
- **Zero imputation.** The write-up's robustness argument assumes a missing modality is
  set to zero **before** z-score normalisation (so it maps to a specific, provably bounded
  offset `-μ/σ`). The current code instead zeroes the modality block **after** scaling, in
  `build_partial_embedding`, which is a simpler but not identical operation — it does not
  reproduce the exact bias term used in the write-up's Proposition 1 derivation.
- **Default configuration.** `VARIANT` defaults to `"without"` (no modality-fair scaling)
  rather than `"with"`, so the modality-fair distance must be explicitly enabled — see
  [Configuration](#configuration).
- **Candidate-set truncation.** The BFS/Dijkstra expansion additionally caps the pool at
  `MAX_NODE2_RL` nodes on top of the depth bound `D`, which isn't part of the formal
  candidate-set definition in the write-up but keeps candidate-set sizes tractable.
- **Oracle definition.** For regression, the oracle is simply the reachable candidate with
  the smallest absolute label difference, rather than requiring an *exact* label match as
  specified for the classification case.

## Citation

```bibtex
@article{rl4m4,
  title   = {RL4M4: Reinforcement Learning for Multi-Modal Missing-Modality Retrieval},
  author  = {<Your Name>},
  year    = {2026},
  note    = {Preprint}
}
```

## License

Add your license of choice here (e.g., MIT, Apache-2.0).
