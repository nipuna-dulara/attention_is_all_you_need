import torch
import numpy as np
print(np.__version__)
print("Hello World!")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
