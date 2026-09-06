import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np
from collections import Counter
import torchaudio
import cv2

def _analyze_mosi_dataset(audio_dir, video_dir, text_dir, split_path):
    """Analyze MOSI dataset to extract statistics"""
    print("\n" + "="*80)
    print("ANALYZING MOSI DATASET (REGRESSION)")
    print("="*80)
    
    with open(split_path, 'r') as f:
        splits = json.load(f)
    
    sample_info = {}
    label_stats = {'train': [], 'val': [], 'test': []}
    
    for split_name in ['train', 'val', 'test']:
        for item in splits[split_name]:
            sample_name = item['name']
            label = item['label']
            sample_info[sample_name] = {
                'label': label,
                'split': split_name
            }
            label_stats[split_name].append(label)
    
    print(f"\nDataset Statistics:")
    print(f"  Split sizes:")
    for split_name in ['train', 'val', 'test']:
        total = len(splits[split_name])
        labels = label_stats[split_name]
        mean_label = np.mean(labels)
        std_label = np.std(labels)
        min_label = np.min(labels)
        max_label = np.max(labels)
        pos_count = sum(1 for l in labels if l >= 0)
        neg_count = sum(1 for l in labels if l < 0)
        
        print(f"    {split_name.capitalize():<5s}: {total:>4d} samples")
        print(f"      Label range: [{min_label:>6.2f}, {max_label:>6.2f}]")
        print(f"      Mean: {mean_label:>6.2f}, Std: {std_label:>6.2f}")
        print(f"      Positive (>=0): {pos_count:>4d} ({100*pos_count/total:>5.1f}%), Negative (<0): {neg_count:>4d} ({100*neg_count/total:>5.1f}%)")
    print("="*80 + "\n")
    
    return splits, sample_info

class MOSIDatasetRegression(Dataset):
    """
    MOSI Dataset with REGRESSION (continuous sentiment scores).
    Label range: [-3, +3]
    """
    def __init__(self, audio_dir, video_dir, text_dir, split_file, split='train',
                 missing_config='100_text_100_audio_100_video',
                 max_text_len=512, seed=42):
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.text_dir = text_dir
        self.split = split
        self.missing_config = missing_config
        self.max_text_len = max_text_len
        self.seed = seed
        
        with open(split_file, 'r') as f:
            splits = json.load(f)
        self.samples = splits[split]
        
        self.parse_missing_config()
        self.modality_availability = self._precompute_modality_availability()
        #self._print_dataset_info()
    
    def parse_missing_config(self):
        """Parse missing configuration string"""
        parts = self.missing_config.split('_')
        
        if parts[0] == 'complex':
            self.complex_mode = True
            self.all_three = int(parts[1]) / 100.0
            self.text_only = int(parts[2]) / 100.0
            self.audio_only = int(parts[3]) / 100.0
            self.video_only = int(parts[4]) / 100.0
            self.text_audio = int(parts[5]) / 100.0
            self.text_video = int(parts[6]) / 100.0
            self.audio_video = int(parts[7]) / 100.0
            
            total = (self.all_three + self.text_only + self.audio_only + 
                    self.video_only + self.text_audio + self.text_video + self.audio_video)
            if abs(total - 1.0) > 0.01:
                raise ValueError(f"Complex ratios must sum to 1.0, got {total}")
        else:
            self.complex_mode = False
            self.text_ratio = int(parts[0]) / 100.0
            self.audio_ratio = int(parts[2]) / 100.0
            self.video_ratio = int(parts[4]) / 100.0
    
    def _precompute_modality_availability(self):
        """Pre-compute modality availability deterministically"""
        rng = np.random.RandomState(self.seed)
        availability = {}
        
        if self.complex_mode:
            for idx in range(len(self.samples)):
                rand_val = rng.random()
                cumsum = 0
                
                if rand_val < (cumsum := cumsum + self.all_three):
                    availability[idx] = (True, True, True)
                elif rand_val < (cumsum := cumsum + self.text_only):
                    availability[idx] = (True, False, False)
                elif rand_val < (cumsum := cumsum + self.audio_only):
                    availability[idx] = (False, True, False)
                elif rand_val < (cumsum := cumsum + self.video_only):
                    availability[idx] = (False, False, True)
                elif rand_val < (cumsum := cumsum + self.text_audio):
                    availability[idx] = (True, True, False)
                elif rand_val < (cumsum := cumsum + self.text_video):
                    availability[idx] = (True, False, True)
                else:
                    availability[idx] = (False, True, True)
        else:
            for idx in range(len(self.samples)):
                has_text = rng.random() < self.text_ratio
                has_audio = rng.random() < self.audio_ratio
                has_video = rng.random() < self.video_ratio
                availability[idx] = (has_text, has_audio, has_video)
        
        return availability
    
    def _print_dataset_info(self):
        """Print dataset statistics"""
        counts = Counter(self.modality_availability.values())
        total = len(self.samples)
        
        # Calculate label statistics
        labels = [s['label'] for s in self.samples]
        mean_label = np.mean(labels)
        std_label = np.std(labels)
        min_label = np.min(labels)
        max_label = np.max(labels)
        pos_count = sum(1 for l in labels if l >= 0)
        neg_count = sum(1 for l in labels if l < 0)
        
        print(f"\n{'='*80}")
        print(f"MOSI Dataset Regression: {self.split} split")
        print(f"  Config: {self.missing_config} | Seed: {self.seed}")
        print(f"  Total samples: {total}")
        print(f"  Label statistics:")
        print(f"    Range: [{min_label:.2f}, {max_label:.2f}]")
        print(f"    Mean: {mean_label:.2f}, Std: {std_label:.2f}")
        print(f"    Positive (>=0): {pos_count} ({100*pos_count/total:.1f}%), Negative (<0): {neg_count} ({100*neg_count/total:.1f}%)")
        print(f"  Modality availability:")
        
        for (t, a, v), count in sorted(counts.items(), key=lambda x: -x[1]):
            modalities = []
            if t: modalities.append("Text")
            if a: modalities.append("Audio")
            if v: modalities.append("Video")
            if not modalities: modalities.append("None")
            print(f"    {'+'.join(modalities):<20s}: {count:>4d} ({100*count/total:>5.1f}%)")
        print(f"{'='*80}\n")
    
    def __len__(self):
        return len(self.samples)
    
    def load_text(self, sample_name):
        """Load text from annotprocessed file"""
        video_name, segment_idx = sample_name.rsplit('_', 1)
        segment_idx = int(segment_idx)
        
        text_file = os.path.join(self.text_dir, f"{video_name}.annotprocessed")
        
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                if line.startswith(f"{segment_idx}_"):
                    text = line.split('_', 1)[1].strip()
                    return text
            return ""
        except:
            return ""
    
    def load_audio(self, sample_name):
        """Load audio waveform"""
        audio_path = os.path.join(self.audio_dir, f"{sample_name}.wav")
        try:
            waveform, sr = torchaudio.load(audio_path)
            return waveform, sr
        except:
            return torch.zeros(1, 16000), 16000
    
    def load_video(self, sample_name):
        """Load video frames"""
        video_path = os.path.join(self.video_dir, f"{sample_name}.mp4")
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
            return frames
        except:
            return []
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_name = sample['name']
        label = torch.tensor(sample['label'], dtype=torch.float32)
        
        has_text, has_audio, has_video = self.modality_availability[idx]
        
        text = self.load_text(sample_name) if has_text else ""
        audio, sr = self.load_audio(sample_name) if has_audio else (torch.zeros(1, 16000), 16000)
        video = self.load_video(sample_name) if has_video else []
        
        if not has_text:
            text = ""
        if not has_audio:
            audio = torch.zeros(1, 16000)
        if not has_video:
            video = []
        
        return {
            'name': sample_name,
            'text': text,
            'audio': audio,
            'video': video,
            'label': label,
            'has_text': has_text,
            'has_audio': has_audio,
            'has_video': has_video
        }


if __name__ == "__main__":
    audio_dir = "/home/office/Downloads/_Dataset/Raw - CMU Multimodal Opinion Sentiment Intensity/Audio/WAV_16000/Segmented"
    video_dir = "/home/office/Downloads/_Dataset/Raw - CMU Multimodal Opinion Sentiment Intensity/Video/Segmented"
    text_dir = "/home/office/Downloads/_Dataset/Raw - CMU Multimodal Opinion Sentiment Intensity/Transcript/Segmented"
    split_file = "/home/office/Downloads/_Dataset/Raw - CMU Multimodal Opinion Sentiment Intensity/mosi_splits-70train.json"
    
    analyze_mosi_dataset(audio_dir, video_dir, text_dir, split_file)
    
    dataset = MOSIDatasetRegression(
        audio_dir=audio_dir,
        video_dir=video_dir,
        text_dir=text_dir,
        split_file=split_file,
        split='train',
        missing_config='complex_20_30_20_10_15_5_0',
        seed=42
    )
    
    print(f"Dataset size: {len(dataset)}\n")
    
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        print(f"Sample {i}:")
        print(f"  Name: {sample['name']}")
        print(f"  Label: {sample['label'].item():.3f} (Sentiment score)")
        print(f"  Text: {sample['text'][:100]}..." if len(sample['text']) > 100 else f"  Text: {sample['text']}")
        print(f"  Has modalities: Text={sample['has_text']}, Audio={sample['has_audio']}, Video={sample['has_video']}")
        print()
