import os
from PIL import Image
import argparse
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset
import wandb
from transformers import ViTForImageClassification

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
    
    
class MyViTDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder(root=root_dir, transform=None)
        self.image_paths = [item[0] for item in self.dataset.samples]  # Extract file paths
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath, label = self.image_paths[idx]
        image = Image.open(filepath).convert("RGB")  # Convert grayscale to RGB
        if self.transform:
            image = self.transform(image)
        return image, label
    



    

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
              'img_norm': args.img_norm}

    assert(num_labels==2) # model applicable for binary classification

    train_dataset = MyViTDataset(os.path.join(args.data_dir, 'train'), transform=vit_transform)
    val_dataset = MyViTDataset(os.path.join(args.data_dir, 'val'), transform=vit_transform)
    test_dataset = MyViTDataset(os.path.join(args.data_dir, 'test'), transform=vit_transform)

    train_dataloader = get_dataloader(train_dataset, CONFIG["batch_size"], train=False)
    val_dataloader = get_dataloader(val_dataset, CONFIG["batch_size"])
    test_dataloader = get_dataloader(test_dataset, CONFIG["batch_size"])

    os.makedirs(args.out_dir, exist_ok=True)

    model_name= f'ViT_{CONFIG["lr"]}_img{CONFIG["img_size"][0]}_b{CONFIG["batch_size"]}'
    best_model_path = os.path.join(args.out_dir, f"{model_name}.pth")


    model = MyViT().to(DEVICE)

    wandb.init(project="compmed", group='ViT')
    wandb.watch(model, log="all", log_freq=10)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-5)

    trainer = Trainer(model, optimizer, criterion, DEVICE)
    
    trainer.train(CONFIG["n_epochs"], train_dataloader, val_dataloader, model_path=best_model_path)

    print("Done!\n")

    trainer.final_evaluate(test_dataloader)



if __name__ == "__main__":
    main()


