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

# Setting the device to cuda
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

with open('configs.pkl', 'rb') as f:
    configs = pickle.load(f)

with open('optimal_config.pkl', 'rb') as f:
    optimal_config = pickle.load(f)

print(f"Optimal Configuration: {optimal_config}")

valid_configs = [config for config in configs if config['val_loss'] is not None and not math.isnan(config['val_loss'])]

if valid_configs:
    min_config = min(valid_configs, key=lambda x: x['val_loss'])
    print(f"Min Configuration: {optimal_config}")
else:
    print("No valid configurations found.")


sample = torch.randn(2**(optimal_config["n"]-7))
print(sample)

opt_vae = torch.load('optimal_vae.pth')
opt_vae.eval()

generated_signature = opt_vae.decoder(sample.to(device))
torch.save(generated_signature, 'opt_model_generated_signature.pt')
