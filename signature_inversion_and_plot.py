import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

other_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Signature'))
sys.path.append(other_folder_path)
from InvertSignatorySignatures import invert_signature

signatures = torch.load('data\signatures.pt')            #loading generated signatures

path_segments = []
paths = []
first_paths =[]

# Iterates through all signatures, inverts them into 10 day paths, makes all paths start at zero, and then appends to the list of paths
for signature in signatures:
    signature = signature.float().unsqueeze(0).cpu()
    inverted_path = invert_signature(signature = signature, depth = 10, channels = 2)[0,1:,:].detach().numpy()
    inverted_path = inverted_path-inverted_path[0,:]                #shifts path to start at zero
    path_segments.append(inverted_path)


for i in range(int(len(path_segments)/3)):
    first_path = path_segments[i*3]
    second_path = path_segments[i*3+1] + path_segments[i*3][-1,:]   #shifting 2nd part of path the begin at end of 1st
    third_path = path_segments[i*3+2] + second_path[-1,:]           #shifting 3rd part of path the begin at end of 2nd
    thirty_day_path = np.concatenate((first_path,second_path, third_path), axis=0)
    paths.append(thirty_day_path)

    first_path=thirty_day_path[:,0]
    first_paths.append(first_path)

num_paths = len(paths)
colors = plt.cm.jet(np.linspace(0,1,num_paths))
for i, path in enumerate(first_paths):
    plt.plot(path, color = colors[i])

plt.show()

# print(path_segments[0].shape) 
# print(paths[0].shape)
