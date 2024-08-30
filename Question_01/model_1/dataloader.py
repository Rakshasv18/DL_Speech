import torch
import torchaudio
from torch.utils.data import Dataset
import numpy as np

class LJSpeechDataset(Dataset):
    def __init__(self, file_paths, transform, mask_percentage=0.2, max_mask_segments=2):
        self.file_paths = file_paths
        self.transform = transform
        self.mask_percentage = mask_percentage
        self.max_mask_segments = max_mask_segments

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        waveform, sr = torchaudio.load(file_path)
        mel_spec = self.transform(waveform)

        # Randomly mask segments of the Mel-spectrogram
        masked_mel_spec, mask = self.random_masking(mel_spec)

        return masked_mel_spec, mel_spec, mask

    def random_masking(self, mel_spec):
        masked_mel_spec = mel_spec.clone()
        mask = torch.zeros_like(mel_spec)

        num_segments = np.random.randint(1, self.max_mask_segments + 1)
        total_mask_len = int(mel_spec.shape[-1] * self.mask_percentage)

        for _ in range(num_segments):
            start = np.random.randint(0, mel_spec.shape[-1] - total_mask_len)
            end = start + total_mask_len
            masked_mel_spec[:, :, start:end] = 0
            mask[:, :, start:end] = 1

        return masked_mel_spec, mask
