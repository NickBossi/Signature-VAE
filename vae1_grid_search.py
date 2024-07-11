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
import itertools

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

def get_mean_std():
    train_sig_data = torch.load('data/train_sig_data.pt')
    column_mean = torch.mean(train_sig_data, dim = 0)
    column_std = torch.std(train_sig_data, dim = 0)
    return column_mean, column_std

# Data prep
data = torch.load(os.path.join('..','Stocks', 'data', 'train_sig_data.pt')).float()                 # Loads data from Stocks folder
column_mean, column_std = get_mean_std()                                                            # Gets mean and std of data
print(column_mean.shape)
print(column_std.shape)
train_signature_data = ((data- column_mean)/column_std).unsqueeze(1)                                # Normalises
input_size = data.shape[2]              # gets dimension of inputs
num_samples = data.shape[0]             # gets number of samples 
print(input_size)
print(num_samples)
train_size = int(0.8*num_samples)       
val_size = num_samples-train_size
dataset = CustomDataset(data)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])      # splitting into training and validation sets

dataloader = DataLoader(dataset=dataset, batch_size=1)


class Encoder(nn.Module):

    def __init__(self, lay1, lay2, lay3, lay4):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(input_size, lay1)                 # First layer
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

# Setting global variables
mse_loss = nn.MSELoss()
global optimal_vae_loss
global optimal_vae_model
global optimal_config
global device
optimal_vae_model = VAE(1,1,1,1)
optimal_config = {}
optimal_vae_loss = 1000000
# Setting the device to cuda
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

def train(config):

    global optimal_vae_loss                 #global variables
    global optimal_vae_model
    global optimal_config

    vae = VAE(2**(config["n"]), 2**(config["n"]-3), 2**(config["n"]-5), 2**(config["n"]-7)).to(device)

    train_load = DataLoader(train_dataset, batch_size = config["batch_size"], shuffle = True, num_workers=0)
    val_load   = DataLoader(val_dataset, batch_size = config["batch_size"], shuffle = True, num_workers=0)

    optimizer  = optim.Adam(vae.parameters(), lr = config["lr"], weight_decay = config["l2_reg"])

    min_val_epoch = 0

    for epoch in range(config["epochs"]):
        train_MSE_loss = 0.0
        train_kl_loss = 0.0
        train_loss = 0.0
        train_steps = 0

        vae.train()         # Sets the model to training mode

        for i, data in enumerate(train_load):
            # Forward pass

            inputs = data.to(device)
            optimizer.zero_grad()
            outputs = vae(inputs).unsqueeze(1).unsqueeze(1)

            MSE_loss = mse_loss(inputs, outputs)
            kl_loss = vae.encoder.kl

            if config["weighting_boolean"]:
                loss = (1-config["kl_weight"])*MSE_loss + config["kl_weight"]*kl_loss
            else:
                loss = MSE_loss + 0.1*epoch*kl_loss

            # Backprop
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)      # Gradient clipping to help prevent gradient blow-up
            optimizer.step()

            # Update loss
            train_loss += loss
            train_MSE_loss += MSE_loss
            train_kl_loss += kl_loss
            train_steps +=1

        # Validation loss
        min_val_loss = 100000000
        val_loss     = 0.0
        val_mse_loss = 0.0
        val_kl_loss = 0.0
        val_steps = 0
        vae.eval()              # sets the model to evaluation mode
        for i, data in enumerate(val_load):
            #print('Validation: {}'.format(val_steps))
            with torch.no_grad():
                inputs = data.to(device)
                outputs = vae(inputs).unsqueeze(1).unsqueeze(1)

                MSE_loss = mse_loss(inputs, outputs)
                kl_loss = vae.encoder.kl

                if config["weighting_boolean"]:
                    print("Got here")
                    loss = (1-config["kl_weight"])*MSE_loss + config["kl_weight"]*kl_loss
                else:
                    loss = MSE_loss + 0.2*epoch*kl_loss
                val_mse_loss += MSE_loss
                val_kl_loss += kl_loss
                val_loss += loss.cpu().numpy()
                val_steps +=1

        print('EPOCH {}. Train MSE loss: {}. Train KL loss: {}. Training loss: {}.\n Valid MSE loss {}. Valid KL loss {}. Validation Loss: {}'.format(epoch+1, 
                                                                                                                                                train_MSE_loss/train_steps, train_kl_loss/train_steps, train_loss/train_steps, val_mse_loss/val_steps, val_kl_loss/val_steps, val_loss/val_steps ))
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            min_val_epoch = epoch
            if val_loss < optimal_vae_loss:
                optimal_vae_loss = val_loss
                optimal_vae_model = vae
                optimal_config = config

    return min_val_loss, min_val_epoch

'''
# Creating table over which we will grid search for hyperparameters
n_list = [8, 12, 14]
lr_list = np.logspace(np.log10(0.1), np.log10(0.0001), num=5)
l2_list = np.logspace(np.log10(0.1), np.log10(0.0001), num=5)
batch_size_list = [1,4,16]
epoch_list = [1]
weighting_boolean_list = [False,True]
kl_weight_list = np.logspace(np.log10(0.5), np.log10(0.005), num=5)
'''


n_list = [8]
lr_list = [0.005]
l2_list = [0]
batch_size_list = [16]
epoch_list = [5]
weighting_boolean_list = [False]
kl_weight_list = []

# dealing with case when we don't have kl vs mse weighting 
param_combinations_no_weight = list(itertools.product(
    n_list,
    lr_list,
    l2_list,
    batch_size_list,
    epoch_list,
))

# Case when we do have kl vs mse weighting
param_combinations_weight = list(itertools.product(
    n_list,
    lr_list,
    l2_list,
    batch_size_list,
    epoch_list,
    kl_weight_list
))

total_number_configs = len(list(param_combinations_weight)) + len(list(param_combinations_no_weight))
running_total = 0
#print(total_number_configs)

configs = []            # empty list in which configurations and their validation losses will be stored

def plot_latent(model, config):
    if 2**(config["n"]-7)==2:

        global device
        latent_embeddings = []
        data = DataLoader(dataset, batch_size=1)

        # Gets encoding of each datapoint (both training and validation)
        for i, input in enumerate(data):
            latent_embedding = model.encoder(input.to(device)).squeeze(0)
            latent_embeddings.append(latent_embedding.detach().cpu().numpy())

        #print("Number samples = {}".format(len(latent_embeddings)))
        # Plots latent space
        x = [embedding[0] for embedding in latent_embeddings]
        y = [embedding[1] for embedding in latent_embeddings]
        plt.scatter(x, y, color = "blue", marker = "o", s = 100)
        plt.title("Latent Space")
        plt.show()

if __name__ == "__main__":

    for (n, lr, l2_reg, batch_size, epochs, kl_weight) in param_combinations_weight:
        config = {
            "n": n,
            "lr": lr,
            "l2_reg": l2_reg,
            "batch_size": batch_size,
            "epochs": epochs,
            "weighting_boolean": True,
            "kl_weight": kl_weight,
            "val_loss": None,
            "min_val_epoch": None
        }
        config["val_loss"], config["min_val_epoch"] = train(config)
        configs.append(config)
        running_total+=1
        print("{} percent done.".format(100*(running_total/total_number_configs)))


    for (n, lr, l2_reg, batch_size, epochs) in param_combinations_no_weight:
        config = {
            "n": n,
            "lr": lr,
            "l2_reg": l2_reg,
            "batch_size": batch_size,
            "epochs": epochs,
            "weighting_boolean": False,
            "kl_weight": None,
            "val_loss": None,
            "min_val_epoch": None
        }
        config["val_loss"], config["min_val_epoch"] = train(config)
        configs.append(config)
        running_total+=1
        print("{} percent done.".format(100*(running_total/total_number_configs)))


    plot_latent(optimal_vae_model, optimal_config)

    torch.save(optimal_vae_model, 'data/optimal_vae.pth')

    with open('data/configs.pkl', 'wb') as f:
        pickle.dump(configs, f)

    with open('data/optimal_config.pkl', 'wb') as f:
        pickle.dump(optimal_config, f)


    


    # opt_vae = torch.load('optimal_vae.pth')
    # sample = torch.randn(2**(optimal_config["n"]-7))
    # generated_signature = opt_vae.decoder(sample.to(device))
    # torch.save(generated_signature, 'opt_model_generated_signature.pt')

# Batch size of one seems best alongside a lower learning rate
# lay1: 1000, lay2: 500 gave worse training but better validation


#vae_model = train(config)

# sample = torch.randn(config["lay4"])
# generated_signature = vae_model.decoder(sample.to(device))
# torch.save(generated_signature, 'generated_signature.pt')


