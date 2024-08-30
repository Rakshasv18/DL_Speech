import torch
import torchaudio
from torch.utils.data import Dataset
import numpy as np

class LJSpeechDataset(Dataset):
    def __init__(self, file_paths, transform, mask_percentage=0.2, max_mask_segments=2):
        """
        Initialize the LJSpeech dataset.

        Args:
            file_paths (list of str): List of file paths to the audio files.
            transform (callable): A function or transform to apply to the waveform.
            mask_percentage (float): Percentage of Mel-spectrogram to mask.
            max_mask_segments (int): Maximum number of non-overlapping segments to mask.
        """
        self.file_paths = file_paths
        self.transform = transform
        self.mask_percentage = mask_percentage
        self.max_mask_segments = max_mask_segments

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns:
            int: Number of samples (file paths) in the dataset.
        """
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (masked_mel_spec, mel_spec, mask) where
                masked_mel_spec (torch.Tensor): Mel-spectrogram with randomly masked segments.
                mel_spec (torch.Tensor): Original Mel-spectrogram.
                mask (torch.Tensor): Binary mask indicating the masked segments.
        """
        file_path = self.file_paths[idx]
        waveform, sr = torchaudio.load(file_path)
        mel_spec = self.transform(waveform)

        # Randomly mask segments of the Mel-spectrogram
        masked_mel_spec, mask = self.random_masking(mel_spec)

        return masked_mel_spec, mel_spec, mask

    def random_masking(self, mel_spec):
        """
        Randomly mask segments of the Mel-spectrogram.

        Args:
            mel_spec (torch.Tensor): Mel-spectrogram of shape (n_channels, n_frames, n_freq_bins).

        Returns:
            tuple: (masked_mel_spec, mask) where
                masked_mel_spec (torch.Tensor): Mel-spectrogram with masked segments.
                mask (torch.Tensor): Binary mask indicating the positions of masked segments.
        """
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
