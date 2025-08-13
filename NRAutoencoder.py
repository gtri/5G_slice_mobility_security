'''
Autoencoder for outlier detection:
- Torch autoencoder model (2 linear layers 50 -> 24 -> 8 -> 24 -> 50)
- Lightning module
'''

import numpy as np 
import argparse
import sys
from pathlib import Path

import torch 
from torch import nn 
from torch import Tensor
import torch.utils.data as data
import lightning as L 
from lightning.pytorch.callbacks import Callback 

from NRDataset import *

def torch_setup():
    # Torch version
    print('Torch: ', torch.__version__)
    # Seed for reporducibility 
    torch.manual_seed(120)
    # GPU availability 
    print('GPU enabled:', torch.cuda.is_available())
    # Set device 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    print('Compute:', device)
    return device

# Naive Autoencoder model
class DSM_Autoencoder(nn.Module):
    '''
    Autoencoder Model  
    '''
    def __init__(self, feat_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feat_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 8),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 24),
            nn.ReLU(),
            nn.Linear(24, feat_dim),
            nn.ReLU()
        )

    def forward(self, in_feats):
        encoded_feats = self.encoder(in_feats)
        decoded_feats = self.decoder(encoded_feats)
        return decoded_feats 

# Lightning module
class LitAutoencoder(L.LightningModule):
    def __init__(self, model, loss_fn):
        super().__init__()
        self.model = model 
        self.loss_fn = loss_fn
        self.reconstruction_loss = []

    def training_step(self, batch, batch_idx):
        x, _ = batch 
        x = x.view(x.size(0), -1)
        x_hat = self.model(x)
        loss = self.loss_fn(x_hat.squeeze(dim=1), x)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss 

    def validation_step(self, batch, batch_idx):
        loss = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_loss": loss}
        self.log('Val loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return metrics

    def test_step(self, batch, batch_idx):
        loss = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_loss": loss}
        self.log('Test loss', loss)
        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self.model(x)
        loss = self.loss_fn(x_hat.squeeze(dim=1), x.float())
        return loss

    def configure_optimizers(self):
        # optimizer = optim.Adam(self.parameters(), lr = 1e-3)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        return optimizer

# Reconstruction Loss Callback
class AutoencoderReconstructionLoss(Callback):
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        x,y = batch
        loss = outputs['test_loss']
        pl_module.reconstruction_loss.append((loss, y))
        

# Reconstruction loss plot 
def plot_reconsctruction_loss(reconstruction_loss): 
    benign_loss_dist = []
    mal_loss_dist = []
    for loss, label in reconstruction_loss:
        if label.item() == 1:
            mal_loss_dist.append(loss)
        else: 
            benign_loss_dist.append(loss)

    fig, ax = plt.subplots(figsize=(6,6))
    ax.hist(benign_loss_dist, bins=20, density=True, label="Benign", alpha=.6)
    ax.hist(mal_loss_dist, bins=20, density=True, label="Malicious", alpha=.6)
    plt.title('Distribution of Reconsturction Loss')
    plt.xlabel('Reconstruction Loss')
    plt.ylabel("Density (calculated by plt.hist)")
    plt.legend()
    plt.savefig('plots/reconstruction-loss.png')


def main(dataset_file: str, dataset_id: int):
    # Run torch setup (seed + device setting)
    device = torch_setup()
    # Data handling 
    datasets, dataloaders = AutoencoderDatasetMain(dataset_file, dataset_id)
    train_dataset, val_dataset, test_dataset = datasets
    train_dataloader, val_dataloader, test_dataloader = dataloaders

    sample, label = datasets[0][1]
    print('Sample data: ', sample, 'Label: ', label)

    # Model definition 
    autoencoder = DSM_Autoencoder(feat_dim = len(sample))
    autoencoder.to(device)
    # Lightning model instantiation
    loss_fn = nn.MSELoss(reduction='mean')
    NR_Autoencoder = LitAutoencoder(autoencoder, loss_fn)
    # Autoencoder training 
    trainer = L.Trainer(limit_train_batches=100, max_epochs=100, callbacks=[AutoencoderReconstructionLoss()])
    trainer.fit(model=NR_Autoencoder, train_dataloaders=train_dataloader) # Model is the lightning module here 
    # Autoencoder test
    trainer.test(NR_Autoencoder, dataloaders=test_dataloader)
    plot_reconsctruction_loss(NR_Autoencoder.reconstruction_loss)


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Train autoencoder and plot reconstruction loss.')
    parser.add_argument('dataset_file', metavar='Dataset Path', type=Path, nargs=1,
                    help='The data file from which to pull.')
    parser.add_argument('dataset_id', metavar='Dataset ID', type=int, nargs=1,
                    help='The type of data for training (Core, slice, or UE).')
    args = parser.parse_args()

    dataset_file = Path(sys.argv[1])
    dataset_id = int(sys.argv[2])

    main(dataset_file=dataset_file, dataset_id=dataset_id)