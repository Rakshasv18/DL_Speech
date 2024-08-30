import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import random
import os

class LJSpeechDataset(Dataset):
    """
    A PyTorch Dataset class for loading and processing LJSpeech audio files.

    Args:
        data_dir (str): Directory containing the audio files.
        mask_percentage (float): Percentage of Mel-spectrogram to be masked.
    """
    
    def __init__(self, data_dir, mask_percentage):
        self.data_dir = data_dir
        self.mask_percentage = mask_percentage
        self.file_list = self._load_files()

    def _load_files(self):
        """
        Load file paths from the dataset directory.

        Returns:
            list of str: List of file paths to the audio files.
        """
        return [f"{self.data_dir}/{file}" for file in os.listdir(self.data_dir)]

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of audio files in the dataset.
        """
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: Masked Mel-spectrogram.
                - torch.Tensor: Original Mel-spectrogram.
                - torch.Tensor: Binary mask used for masking.
        """
        file_path = self.file_list[idx]
        waveform, sample_rate = torchaudio.load(file_path)
        mel_spectrogram = torchaudio.transforms.MelSpectrogram()(waveform)
        masked_mel, mask = self._apply_mask(mel_spectrogram)
        return masked_mel, mel_spectrogram, mask

    def _apply_mask(self, mel_spectrogram):
        """
        Apply a random mask to the Mel-spectrogram.

        Args:
            mel_spectrogram (torch.Tensor): Original Mel-spectrogram to be masked.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: Masked Mel-spectrogram.
                - torch.Tensor: Binary mask indicating the masked regions.
        """
        length = mel_spectrogram.shape[-1]
        mask_len = int(length * random.choice(self.mask_percentage))
        start = random.randint(0, length - mask_len)
        mask = torch.ones_like(mel_spectrogram)
        mask[:, start:start + mask_len] = 0
        masked_mel = mel_spectrogram * mask
        return masked_mel, mask

def get_data_loader(data_dir, batch_size, mask_percentage):
    """
    Create a DataLoader for the LJSpeech dataset.

    Args:
        data_dir (str): Directory containing the audio files.
        batch_size (int): Number of samples per batch.
        mask_percentage (float): Percentage of Mel-spectrogram to be masked.

    Returns:
        DataLoader: DataLoader instance for the LJSpeech dataset.
    """
    dataset = LJSpeechDataset(data_dir, mask_percentage)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
