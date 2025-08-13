'''
Standalone file for time series dataset creation and AI model 

NRTimeDataset
ConvAutoencoder - Torch model
LitTimeAutoencoder - Time sensitive autoencoder model 
'''
import numpy as np 
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt 
import math

# Data loader imports
import pandas as pd
import torch 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn import preprocessing
from sklearn.pipeline import Pipeline

# AI/ML Imports
from torch import nn 
from torch import Tensor
import lightning as L 
from lightning.pytorch.callbacks import Callback 


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


# Process samples into frame 
def process_raw_features(raw_features: pd.DataFrame, transform, raw_labels: pd.DataFrame = None, frame_length: int = 12):
    '''
    Takes raw samples, applies normalization, and stacks them into frames

    raw_samples: df of just samples - no labels 
    frame_length: Number of samples that go into a frame
    transform: The normalization and scaling pipeline (should already be fit to data)
    '''
    # Apply transformation to raw data
    features = torch.tensor(transform.transform(raw_features)).to(torch.float32)

    # Iterate over samples to 
    sample_frame = []
    timestamps = []
    snssai = []
    labels = []
    for i in range(len(raw_features.index) - frame_length):
        # If there is a huge gap in time - skip samples and jump indicies to after the gap 
        if (abs(raw_features.index[(i+frame_length)] - raw_features.index[i]) > 200):
            pass
        # Create a frame of frame_length
        else: 
            sample_frame.append(features[i:(i+frame_length)])
            timestamps.append(raw_features.index[i])
            # If no labels are provided an empty list will be returned 
            if raw_labels is None:
                pass
            # If labels are given take the value at the start of a frame as the truth value 
            # (By skipping jumps in time we ensure that all the samples in a given frame will have the same label)
            else: 
                labels.append(raw_labels.values[i])

    return sample_frame, timestamps, labels


# TODO: Fix labeling and set a mechanicsm to create samples only based on sequences of data that are from the same time 
# Time series dataset 
class NRTimeDataset(Dataset):
    def __init__(self, df: pd.DataFrame = None, dataset_file: Path = None, frame_length: int = 24, transform=None, dataset_id: int = 1, fit: bool = True): 
        '''
        Loads data, applies transforms, and shapes into time frames.

        Vars: 
        Use either df or dataset_file (will use the df if given both)
        frame_length: Number of samples to put in a stack 
        dataset_id - 0: Core Data, 1: Slice data, 2: UE data (needed to shape labels consistently)
        '''
        # Load data
        if df is not None: 
            self.raw_data = df.set_index('timestamp')
        elif dataset_file is not None: 
            self.raw_data = pd.read_csv(dataset_file).set_index('timestamp')

        # Load raw samples 
        if dataset_id == 0: # Core dataset
            self.raw_samples = self.raw_data.iloc[:, 0:-1]
            self.raw_labels = self.raw_data.iloc[:, -1]
        elif dataset_id == 1: # Slice dataset
            self.raw_samples = self.raw_data.iloc[:,0:-2]
            self.raw_labels = self.raw_data.iloc[:, -2]
            self.snssai = self.raw_data.loc[:, 'slice_id']
        elif dataset_id == 2: # UE dataset
            self.raw_samples = self.raw_data.iloc[:, 1:-2]
            self.raw_labels = self.raw_data.iloc[:, -1]

        # Apply normalization / scaling if specified 
        if fit: 
            self.transform = transform.fit(self.raw_samples)

        sample_frame, timestamps, labels = process_raw_features(self.raw_samples, 
                                                                transform, 
                                                                self.raw_labels, 
                                                                frame_length)

        self.samples = torch.stack(sample_frame).to(torch.float32)
        self.labels = labels
        self.timestamps = timestamps

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.labels[idx]
        return sample, label

    def get_timestamp(self, idx):
        return self.timestamps[idx]
    

# Time-dependent Autoencoder model
class ConvAutoencoder(nn.Module):
    '''
    1D Conv Autoencoder Model
    Takes in ~1min of samples at a time 
    Input shape: (training_size, n_samples, n_features)
    1D Convs
    Input channels = n features 
    Out Channels = n filters  
    '''
    def __init__(self, n_features, filter_sizes: list = [24, 8]):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(n_features,filter_sizes[0],kernel_size=3,padding=2),
            nn.ReLU(),
            nn.Conv1d(filter_sizes[0],filter_sizes[1],kernel_size=3,padding=2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(filter_sizes[1],filter_sizes[0],kernel_size=3,padding=2),
            nn.ReLU(),
            nn.ConvTranspose1d(filter_sizes[0],n_features,kernel_size=3,padding=2),
            nn.ReLU()
        )

    def forward(self, in_feats):
        # in_feats: (Batch size, n_samples, n_features)
        encoded_feats = self.encoder(in_feats)
        decoded_feats = self.decoder(encoded_feats)
        return decoded_feats 


# Lightning module
class LitTimeAutoencoder(L.LightningModule):
    def __init__(self, model, loss_fn):
        super().__init__()
        self.model = model 
        self.loss_fn = loss_fn
        self.reconstruction_loss = []

    def training_step(self, batch, batch_idx):
        x, _ = batch 
        x_hat = self.model(x)
        loss = self.loss_fn(x_hat, x)
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
        loss = self.loss_fn(x_hat, x.float())
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
    plt.title('Distribution of Reconstruction Loss')
    plt.xlabel('Reconstruction Loss')
    plt.ylabel("Density (calculated by plt.hist)")
    plt.legend()
    plt.savefig('plots/reconstruction-loss.png')


# Calc mean and std 
def loss_statistics(reconstruction_loss):
    mean = sum(reconstruction_loss) / len(reconstruction_loss)
    std = np.sqrt(sum([(reconstruction_loss - mean)**2 for reconstruction_loss in reconstruction_loss]) / len(reconstruction_loss))
    return mean, std


# Mahalanobis Distance
def sample_distance(loss, mean, std):
    '''
    The m-distance calculation is really basic because we're doing a 
    comparison to loss not an entire sample vector 
    '''
    m_dist = abs(loss - mean) / std
    return m_dist


# Sample classification 
def sample_classification(sample, benign_mean, benign_std, model, loss_fn, threshold):
    # Reconstruct sample and calculate distance of reconstruction loss from benign distribution 
    x = sample 
    x_hat = model(x)
    loss = loss_fn(x_hat, x.float())
    # m_dist = sample_distance(loss, benign_mean, benign_std)
    
    # If greater than threshold set to malicious else benign
    if loss > threshold:
        y = 1
    else:
        y = 0
    return y


# Identify samples of potentially malicious UEs 
def id_ues(t_attack, snssai, ue_df, transform, frame_size):
    '''
    After detecting and attack we identify which UEs were active on that slice 
    '''
    snssai_to_ue_tag = {'1-111111': 'slice1_count', 
    '1-222222': 'slice2_count', 
    '2-333333': 'slice3_count', 
    '2-444444': 'slice4_count', 
    '3-555555': 'slice5_count', 
    '3-666666': 'slice6_count'
    }

    ue_mask = (ue_df['timestamp'] >= int(t_attack)-3) & (ue_df['timestamp'] <= int(t_attack)+3)
    candidate_samples = ue_df.loc[ue_mask]
    candidate_samples = candidate_samples.loc[(candidate_samples[snssai_to_ue_tag[snssai]]>0)]

    # Exit out if there are no corresponding timestamps 
    if len(candidate_samples) == 0:
        return None

    # Convert the raw UE data samples into time frames 
    all_samples = torch.tensor(transform.transform(ue_df.iloc[:, 1:-2])).to(torch.float32) 
    all_imsis = ue_df.iloc[:, -2].values
    candidate_frames = []
    candidate_imsis = []
    for idx in candidate_samples.index:
        candidate_frames.append(all_samples[idx:(idx+frame_size)])
        candidate_imsis.append(all_imsis[idx])

    candidate_frames = torch.stack(candidate_frames).to(torch.float32)

    candidates = [(sample, imsi) for sample, imsi in zip(candidate_frames, candidate_imsis)]

    return candidates


def correlate_ues(candidates, benign_mean, benign_std, model, loss_fn, threshold):
    mal_ues = []
    for frame, imsi in candidates:
        y = sample_classification(frame, benign_mean, benign_std, model, loss_fn, threshold)
        if y == 1: 
            mal_ues.append(imsi)

    return mal_ues

