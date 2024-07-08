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

import ray
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle

from sklearn.model_selection import KFold

import wandb

CPU_cores = 4

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

        # Normalizing the input data
        x = (x-self.mean)/self.std
        
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
        x = F.relu(self.linear1(x))
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
    
def train_cifar(config, datadir = None):

    vae = VAE(config["lay1"], config["lay2"], config["lay3"], config["lay4"]).to(device)          

    optimizer  = optim.Adam(vae.parameters(), lr = config["lr"], weight_decay = config["l2_reg"])

    checkpoint = get_checkpoint()
    if checkpoint: 
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            vae.load_state_dict(checkpoint_state["net_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    
    else:
        start_epoch = 0

    
    train_load = DataLoader(train_dataset, batch_size = config["batch_size"], shuffle = True, num_workers=0)
    val_load   = DataLoader(val_dataset, batch_size = config["batch_size"], shuffle = True, num_workers=0)

    for epoch in range(start_epoch, config["epochs"]):
        train_loss = 0.0
        train_steps = 0

        vae.train()         #sets the model to training mode

        for i, data in enumerate(train_load):
            #print('Train: {}'.format(train_steps))
            # Forward pass
            inputs = data.to(device)
            optimizer.zero_grad()
            outputs = vae(inputs)

            #loss = ((inputs-outputs)**2).sum() + vae.encoder.kl                
            if config["weighting_boolean"]:
                loss = (1-config["kl_weight"])*((inputs-outputs)**2).sum() + config["kl_weight"]*vae.encoder.kl
            else:
                loss = ((inputs-outputs)**2).sum() + vae.encoder.kl

            # Backprop
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

            # Update loss
            train_loss += loss
            train_steps +=1

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        vae.eval()              # sets the model to evaluation mode
        for i, data in enumerate(val_load):
            #print('Validation: {}'.format(val_steps))
            with torch.no_grad():
                inputs = data.to(device)
                outputs = vae(inputs)
                if config["weighting_boolean"]:
                    loss = (1-config["kl_weight"])*((inputs-outputs)**2).sum() + config["kl_weight"]*vae.encoder.kl
                else:
                    loss = ((inputs-outputs)**2).sum() + vae.encoder.kl
                val_loss += loss.cpu().numpy()
                val_steps +=1
        
        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": vae.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "wb") as fp:
                pickle.dump(checkpoint_data, fp)

            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            train.report(
                {"loss": val_loss / val_steps},
                checkpoint=checkpoint
            )

        print('EPOCH {}. Training loss: {}. Validation Loss: {}'.format(epoch+1, train_loss/train_steps, val_loss/val_steps ))
  
#    return vae

def main(num_samples = 10, max_num_epochs = 10, gpus_per_trial = 1):
    data_dir = os.path.abspath("./data")

    
    config = {
        "lay1": tune.choice([2**i for i in range(12,15)]),
        "lay2": tune.choice([2**i for i in range(10,12)]),
        "lay3": tune.choice([2**i for i in range(8,10)]),
        "lay4": tune.choice([2**i for i in range(6,8)]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "l2_reg": tune.loguniform(1e-4, 1e-1),
        "epochs": 10,
        "batch_size": tune.choice([2,4,8,16]),
        "weighting_boolean": tune.choice([True, False]),
        "kl_weight": tune.choice([0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8])
    }

    scheduler = ASHAScheduler(
        metric = "loss",
        mode = "min",
        max_t = max_num_epochs,
        grace_period=1,
        reduction_factor=2
    )
    ray.init()
    result = tune.run(
        partial(train_cifar, data_dir = data_dir),
        resources_per_trial={"cpu": CPU_cores, "gpu": gpus_per_trial},
        config = config,
        num_samples=num_samples,
        scheduler = scheduler
    )

    
    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")

    best_trained_model = VAE(best_trial.config["lay1"], best_trial.config["lay2"], best_trial.config["lay3"], best_trial.config["lay4"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint = result.get_best_checkpoint(trial=best_trial, metric="loss", mode="min")
    with best_checkpoint.as_directory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "rb") as fp:
            best_checkpoint_data = pickle.load(fp)

        best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])

if __name__ == "__main__":
    main(num_samples = 10, max_num_epochs=10, gpus_per_trial=1)
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


# vae_model = train(config, datadir = None) 

# sample = torch.randn(config["lay4"])
# generated_signature = vae_model.decoder(sample.to(device))
# torch.save(generated_signature, 'generated_signature.pt')


