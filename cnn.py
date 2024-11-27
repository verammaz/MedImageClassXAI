import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import pandas as pd
import argparse
import wandb
from PIL import Image as PILImage

wandb.login()

# Global Variables
img_size = (256, 256)
num_labels = 2
lablel_to_str = {0: "normal", 1: "pneumonia"}

# Get device for training
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

