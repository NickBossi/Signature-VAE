import sys
import os
import torch
import matplotlib.pyplot as plt

other_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Signature'))
sys.path.append(other_folder_path)
from InvertSignatorySignatures import invert_signature

data = torch.load('opt_model_generated_signature.pt').float().unsqueeze(0).cpu()
print(data[0,0])
inverse = invert_signature(signature = data, depth = 10, channels = 2)[0,1:,:].detach().numpy()

plt.plot(inverse)
plt.show()

