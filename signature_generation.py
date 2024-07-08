from functools import partial
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
import itertools
import pickle
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import math
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from vae1_grid_search import Encoder, Decoder, VAE

num_paths = 100             # number of paths to be generated

signatures = []             #list to store generated signatures

# Setting the device to cuda
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Loading the optimal hyperparameters found 
with open('optimal_config.pkl', 'rb') as f:
    optimal_config = pickle.load(f)

# Loading the optimal model found
final_vae = torch.load('optimal_vae.pth')
final_vae.eval()

# Each signatures is 10 days, so need three signatures per full 30 day sample path
for i in range(3*num_paths):
    mu = np.array([0,0])
    std_devs = np.array([5,5])
    sigma = np.diag(std_devs**2)
    sample = (torch.tensor((np.random.multivariate_normal(mu, sigma, 1)))).float().squeeze(0)
    #sample = torch.randn(2**(optimal_config["n"]-7))
    generated_signature = final_vae.decoder(sample.to(device))
    signatures.append(generated_signature)

torch.save(signatures, 'signatures.pt')

