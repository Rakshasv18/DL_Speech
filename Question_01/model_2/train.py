import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import wandb
import yaml
from model import MaskedAutoencoder  
from dataloader import get_data_loader

def train_model(data_loader, config):
    """
    Train a Masked Autoencoder model.

    Args:
        data_loader (DataLoader): DataLoader instance for loading the dataset.
        config (dict): Configuration dictionary containing hyperparameters and training settings. Keys include:
            - 'learning_rate' (float): Learning rate for the optimizer.
            - 'num_epochs' (int): Number of training epochs.
            - 'log_interval' (int): Interval for logging training progress.
    """
    # Initialize model, loss function, optimizer, and logging
    model = MaskedAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    writer = SummaryWriter()
    wandb.init(project="masked-infill")

    for epoch in range(config['num_epochs']):
        for i, (masked_mel, mel_spectrogram, mask) in enumerate(data_loader):
            masked_mel, mel_spectrogram = masked_mel.to(device), mel_spectrogram.to(device)
            optimizer.zero_grad()
            output = model(masked_mel)
            loss = criterion(output * mask, mel_spectrogram * mask)
            loss.backward()
            optimizer.step()

            if i % config['log_interval'] == 0:
                print(f"Epoch [{epoch}/{config['num_epochs']}], Step [{i}/{len(data_loader)}], Loss: {loss.item():.4f}")
                writer.add_scalar('Loss/train', loss.item(), epoch * len(data_loader) + i)
                wandb.log({"loss": loss.item()})

    # Close TensorBoard writer and WandB logger
    writer.close()
    wandb.finish()

# Load configuration
with open("config.yaml", 'r') as file:
    config = yaml.safe_load(file)

# Initialize data loader and train
data_loader = get_data_loader("path/to/LJSpeech", config['batch_size'], config['mask_percentage'])
train_model(data_loader, config)
