from functools import partial
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import random
import tempfile
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import time
from torch.utils.data import Dataset, DataLoader, random_split
import copy
import numpy as np

from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle

from sklearn.model_selection import KFold


# Setting the device to cuda
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Creating custom data set
class CustomDataset(Dataset):
    def __init__(self, input_tensor, transform = None):
        self.input_tensor = input_tensor
        self.mean = self.input_tensor.mean(dim = 0, keepdim = True)
        self.std = self.input_tensor.std(dim = 0, keepdim = True)
        self.transform = transform

    def __len__(self):
        return(len(self.input_tensor))

    def __getitem__(self, index):

        x = self.input_tensor[index]
        
        return x

# Data prep
data = torch.load('train_sig_data.pt').float()
input_size = data.shape[2]
num_samples = data.shape[0]
train_size = int(0.8*num_samples)
val_size = num_samples-train_size
dataset = CustomDataset(data)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

class Encoder(nn.Module):

    def __init__(self, lay1, lay2, lay3, lay4):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(input_size, lay1)              # First layer
        self.linear2 = nn.Linear(lay1, lay2)
        self.linear3 = nn.Linear(lay2, lay3)
        self.linearmean = nn.Linear(lay3, lay4)                    # Mean latent layer
        self.linearvar = nn.Linear(lay3, lay4)                    # Variance latenet layer

        self.N = torch.distributions.Normal(0,1)
        self.N.loc = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = (self.linear1(x))
        x = F.relu(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        mu    = self.linearmean(x)
        sigma = torch.exp(self.linearvar(x)) +1e-6
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma)-1/2).sum()
        return z

class Decoder(nn.Module):

    def __init__(self, lay1, lay2, lay3, lay4):
        super(Decoder, self).__init__()
        self.linear1 =  nn.Linear(lay4, lay3)
        self.linear2 = nn.Linear(lay3,lay2)
        self.linear3 = nn.Linear(lay2, lay1)
        self.linear4 = nn.Linear(lay1, input_size)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        z = F.relu(self.linear3(z))
        z = self.linear4(z)
        return z

class VAE(nn.Module):

    def __init__(self, lay1, lay2, lay3, lay4):
        super(VAE, self).__init__()
        self.encoder = Encoder(lay1, lay2, lay3, lay4)
        self.decoder = Decoder(lay1, lay2, lay3, lay4)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

mse_loss = nn.MSELoss()

def train(config):

    vae = VAE(config["lay1"], config["lay2"], config["lay3"], config["lay4"]).to(device)

    train_load = DataLoader(train_dataset, batch_size = config["batch_size"], shuffle = True, num_workers=0)
    val_load   = DataLoader(val_dataset, batch_size = config["batch_size"], shuffle = True, num_workers=0)

    optimizer  = optim.Adam(vae.parameters(), lr = config["lr"], weight_decay = config["l2_reg"])

    for epoch in range(config["epochs"]):
        train_MSE_loss = 0.0
        train_kl_loss = 0.0
        train_loss = 0.0
        train_steps = 0

        vae.train()         #sets the model to training mode

        for i, data in enumerate(train_load):
            #print(i)
            #print('Train: {}'.format(train_steps))
            # Forward pass
            inputs = data.to(device).squeeze(0).squeeze(0)
            optimizer.zero_grad()
            outputs = vae(inputs)

            #loss = ((inputs-outputs)**2).sum() + vae.encoder.kl
            MSE_loss = mse_loss(inputs, outputs)
            kl_loss = vae.encoder.kl

            if config["weighting_boolean"]:
                loss = (1-config["kl_weight"])*MSE_loss #+ config["kl_weight"]*kl_loss
            else:
                loss = MSE_loss + kl_loss

            # Backprop
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

            # Update loss
            train_loss += loss
            train_MSE_loss += MSE_loss
            train_kl_loss += kl_loss
            train_steps +=1

        # Validation loss
        val_loss = 0.0
        val_mse_loss = 0.0
        val_kl_loss = 0.0
        val_steps = 0
        vae.eval()              # sets the model to evaluation mode
        for i, data in enumerate(val_load):
            #print('Validation: {}'.format(val_steps))
            with torch.no_grad():
                inputs = data.to(device).squeeze(0).squeeze(0)
                outputs = vae(inputs)

                MSE_loss = mse_loss(inputs, outputs)
                kl_loss = vae.encoder.kl

                if config["weighting_boolean"]:
                    loss = (1-config["kl_weight"])*MSE_loss + config["kl_weight"]*kl_loss
                else:
                    loss = MSE_loss + kl_loss
                val_mse_loss += MSE_loss
                val_kl_loss += kl_loss
                val_loss += loss.cpu().numpy()
                val_steps +=1

        print('EPOCH {}. Train MSE loss: {}. Train KL loss: {}. Training loss: {}.\n Valid MSE loss {}. Valid KL loss {}. Validation Loss: {}'.format(epoch+1, 
                                                                                                                                                train_MSE_loss/train_steps, train_kl_loss/train_steps, train_loss/train_steps, val_mse_loss/val_steps, val_kl_loss/val_steps, val_loss/val_steps ))

    return vae
n=12

config = {
    "lay1": 2**n,
    "lay2": 2**(n-3),
    "lay3":2**(n-5),
    "lay4":2**(n-7),
    "lr": 3e-4,
    "l2_reg": 0.0,
    "batch_size": 1,
    "epochs": 10,
    "weighting_boolean": True,
    "kl_weight": 0.2
}
# config = {
#     "lay1": 1000,
#     "lay2": 500,
#     "lay3":125,
#     "lay4":20,
#     "lr": 3e-4,
#     "l2_reg": 0.01,
#     "batch_size": 1,
#     "epochs": 10,
#     "weighting_boolean": False,
#     "kl_weight": 0.2
# }

# Batch size of one seems best alongside a lower learning rate
# lay1: 1000, lay2: 500 gave worse training but better validation


vae_model = train(config)

# sample = torch.randn(config["lay4"])
# generated_signature = vae_model.decoder(sample.to(device))
# torch.save(generated_signature, 'generated_signature.pt')


