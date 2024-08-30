import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchaudio.transforms as T
from model import UNet
from dataloader import LJSpeechDataset
import yaml
import os
import glob
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-length Mel-spectrograms by padding or truncating.

    Args:
        batch (list of tuples): A batch of tuples, each containing (masked_mel_spec, mel_spec, mask).

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: Padded or truncated masked Mel-spectrograms.
            - torch.Tensor: Padded or truncated original Mel-spectrograms.
            - torch.Tensor: Padded or truncated masks.
    """
    masked_mel_specs, mel_specs, masks = zip(*batch)

    # Find the maximum length in this batch
    max_len = max([mel_spec.size(2) for mel_spec in mel_specs])

    # Pad or truncate all tensors to the maximum length
    padded_masked_mel_specs = [F.pad(mel_spec, (0, max_len - mel_spec.size(2))) if mel_spec.size(2) < max_len else mel_spec[:, :, :max_len] for mel_spec in masked_mel_specs]
    padded_mel_specs = [F.pad(mel_spec, (0, max_len - mel_spec.size(2))) if mel_spec.size(2) < max_len else mel_spec[:, :, :max_len] for mel_spec in mel_specs]
    padded_masks = [F.pad(mask, (0, max_len - mask.size(2))) if mask.size(2) < max_len else mask[:, :, :max_len] for mask in masks]

    return torch.stack(padded_masked_mel_specs), torch.stack(padded_mel_specs), torch.stack(padded_masks)

# Load config
with open("config.yaml", "r") as file:
    """
    Load configuration settings from a YAML file.

    Args:
        file (file object): File object for the YAML configuration file.
    
    Returns:
        dict: Configuration settings loaded from the YAML file.
    """
    config = yaml.safe_load(file)

# Initialize TensorBoard SummaryWriter
writer = SummaryWriter(log_dir="runs/experiment_1")
"""
Initialize TensorBoard SummaryWriter to log training metrics.

Args:
    log_dir (str): Directory where TensorBoard logs will be saved.
"""

# Mel-spectrogram transformation
transform = T.MelSpectrogram(
    sample_rate=config["mel_spectrogram_params"]["sample_rate"],
    n_fft=config["mel_spectrogram_params"]["n_fft"],
    hop_length=config["mel_spectrogram_params"]["hop_length"],
    n_mels=config["mel_spectrogram_params"]["n_mels"]
)
"""
Initialize Mel-spectrogram transformation based on configuration parameters.

Args:
    sample_rate (int): Sample rate of the audio.
    n_fft (int): Number of FFT points.
    hop_length (int): Number of audio frames between STFT columns.
    n_mels (int): Number of Mel bands to generate.
"""

# Dataset and Dataloader
dataset = LJSpeechDataset(
    file_paths=glob.glob(os.path.join("/Users/raksha/Downloads/Flawless/audio/LJSpeech-1.1/wavs", "*.wav")),
    transform=transform,
    mask_percentage=config["mask_percentage"],
    max_mask_segments=config["max_mask_segments"]
)
"""
Initialize LJSpeechDataset with file paths, transformation, and masking parameters.

Args:
    file_paths (list of str): List of paths to audio files.
    transform (callable): Function to transform the waveform to Mel-spectrogram.
    mask_percentage (float): Percentage of Mel-spectrogram to mask.
    max_mask_segments (int): Maximum number of non-overlapping segments to mask.
"""

# Update DataLoader to use the custom collate function
dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=custom_collate_fn)
"""
Initialize DataLoader with custom collate function to handle variable-length Mel-spectrograms.

Args:
    dataset (Dataset): Dataset object.
    batch_size (int): Number of samples per batch.
    shuffle (bool): Whether to shuffle data at every epoch.
    collate_fn (callable): Custom collate function to pad or truncate variable-length tensors.
"""

# Model, Optimizer, and Losses
model = UNet(in_channels=1, out_channels=1)
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
mse_loss = nn.MSELoss()
"""
Initialize UNet model, optimizer, and loss function.

Args:
    in_channels (int): Number of input channels.
    out_channels (int): Number of output channels.
    learning_rate (float): Learning rate for the optimizer.
"""

def smoothness_loss(masked_mel_spec, reconstructed_spec, mask):
    """
    Compute smoothness loss based on the difference between the masked Mel-spectrogram and the reconstructed spectrogram.

    Args:
        masked_mel_spec (torch.Tensor): Masked Mel-spectrogram.
        reconstructed_spec (torch.Tensor): Reconstructed Mel-spectrogram.
        mask (torch.Tensor): Binary mask indicating the masked regions.

    Returns:
        torch.Tensor: Smoothness loss value.
    """
    diff = torch.abs(reconstructed_spec - masked_mel_spec)
    smoothness = torch.mean(diff * mask)
    return smoothness

# Training Loop
model.train()
for epoch in range(config["num_epochs"]):
    """
    Training loop for the UNet model.

    Args:
        epoch (int): Current epoch number.
    """
    total_loss = 0
    for masked_mel_spec, mel_spec, mask in dataloader:
        optimizer.zero_grad()
        reconstructed_spec = model(masked_mel_spec)
        
        # Loss calculation
        loss = mse_loss(reconstructed_spec, mel_spec) + smoothness_loss(mel_spec, reconstructed_spec, mask)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch + 1}/{config['num_epochs']}], Loss: {avg_loss:.4f}")

    # Log average loss to TensorBoard
    writer.add_scalar("Loss/train", avg_loss, epoch)

# Close the writer
writer.close()
"""
Close the TensorBoard writer after training is complete.
"""
