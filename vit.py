import os
from PIL import Image
import argparse
import torch
from torch import nn
from torch import optim
import wandb
from transformers import ViTForImageClassification, ViTImageProcessor

from train import Trainer
from utils import *



wandb.login()

# Global Variables
num_labels = 2
lablel_to_str = {0: "normal", 1: "pneumonia"}

# Get device for training
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Model configuration
CONFIG = {}

class MyViT(nn.Module):
    def __init__(self):
        super(MyViT, self).__init__()
        self.model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",  # Pretrained on ImageNet-21k
            num_labels=2  # Binary classification for pneumonia detection
        )

    def forward(self, x):
        return self.model(x)
    
    
class PneumoniaDataset(Dataset):
    def __init__(self, root_dir, processor, transform=None):
        self.root_dir = root_dir
        self.classes = sorted([cls for cls in os.listdir(root_dir) if cls not in [".DS_Store"]])
        self.filepaths = [
            (os.path.join(root, fname), label)
            for label, cls in enumerate(self.classes)
            for root, _, files in os.walk(os.path.join(root_dir, cls))
            for fname in files if fname.endswith(".png") or fname.endswith(".jpeg")
        ]
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath, label = self.filepaths[idx]
        assert label in [0, 1], f"Invalid label {label} in dataset!"  # Add sanity check

        image = Image.open(filepath).convert("RGB")  # Convert grayscale to RGB
        if self.transform:
            image = self.transform(image)
        return image, label
    


# Define the processor for ViT
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")


# Transform function for PyTorch DataLoader
def transform(image):
    # Use the Hugging Face processor to preprocess images
    processed = processor(image, return_tensors="pt")
    pixel_values = processed["pixel_values"].squeeze(0)  # Remove batch dimension
    return pixel_values.contiguous()  # Ensure the tensor is contiguous
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', required=True, help='path to datasets')
    parser.add_argument('-v', action='store_true', help='verbose mode')
    parser.add_argument('-out_dir', default='out', help='path where to save trained model parameters')
    parser.add_argument('-batch_size', default=32)
    parser.add_argument('-n_epochs', default=8)
    parser.add_argument('-lr', default=0.001)
    parser.add_argument('-img_channels', default=1)
    parser.add_argument('-img_norm', action='store_true')

    args = parser.parse_args()

    CONFIG = {'batch_size': int(args.batch_size),
              'n_epochs': int(args.n_epochs),
              'lr': float(args.lr),
              'img_norm': args.img_norm,}

    assert(num_labels==2) # model applicable for binary classification

    train_dataset = PneumoniaDataset(os.path.join(args.data_dir, 'train'), processor, transform=transform)
    val_dataset = PneumoniaDataset(os.path.join(args.data_dir, 'val'), processor, transform=transform)
    test_dataset = PneumoniaDataset(os.path.join(args.data_dir, 'test'), processor, transform=transform)

    train_dataloader = get_dataloader(train_dataset, CONFIG["batch_size"], train=False)
    val_dataloader = get_dataloader(val_dataset, CONFIG["batch_size"])
    test_dataloader = get_dataloader(test_dataset, CONFIG["batch_size"])


    model = MyViT().to(DEVICE)

    wandb.init(project="compmed", group='SimpleCNN')
    wandb.watch(model, log="all", log_freq=10)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-5)

    trainer = Trainer(model, optimizer, criterion, DEVICE)
    
    trainer.train(CONFIG["n_epochs"], train_dataloader, val_dataloader, model_path=best_model_path)

    print("Done!\n")

    trainer.final_evaluate(test_dataloader)



if __name__ == "__main__":
    main()


