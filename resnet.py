import os
import argparse
import torch
from torch import nn
from torch import optim
from torchvision.models import resnet18
import wandb

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

class MyResNET(nn.Module):
    def __init__(self, freeze_params=True):
        super(MyResNET, self).__init__()
        self.model = resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(1, self.model.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_labels)

        if freeze_params: self.freeze_params()

    def freeze_params(self):

        for param in self.model.parameters():
            param.requires_grad = False  # Freeze all layers

        self.model.fc.weight.requires_grad = True  # Unfreeze final layer
        self.model.fc.bias.requires_grad = True
        self.model.conv1.weight.requires_grad = True # Unfreeze first layer

    def forward(self, x):
        return self.model(x)
    



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', required=True, help='path to datasets')
    parser.add_argument('-out_dir', default='out', help='path where to save trained model parameters')
    parser.add_argument('-batch_size', default=64)
    parser.add_argument('-n_epochs', default=8)
    parser.add_argument('-lr', default=0.001)
    parser.add_argument('-img_size', default=256)
    parser.add_argument('-img_norm', action='store_true')
    parser.add_argument('-freeze_params', action='store_true')
    parser.add_argument('-tta', action='store_true', help='whether or not to do test time augmentation')



    args = parser.parse_args()

    CONFIG = {'batch_size': int(args.batch_size),
              'n_epochs': int(args.n_epochs),
              'lr': float(args.lr),
              'img_size': (int(args.img_size), int(args.img_size)),
              'img_norm': args.img_norm}

    assert(num_labels==2) # model applicable for binary classification

    
    train_transforms = get_train_transform(CONFIG['img_size'], CONFIG['img_norm'])

    common_transforms = get_common_transform(CONFIG['img_size']) if not args.tta else train_transforms

    train_dataset = get_dataset(args.data_dir, 'train', train_transforms)
    val_dataset = get_dataset(args.data_dir, 'val', common_transforms)
    test_dataset = get_dataset(args.data_dir, 'test', common_transforms)

    train_dataloader = get_dataloader(train_dataset, CONFIG["batch_size"], train=True)
    val_dataloader = get_dataloader(val_dataset, CONFIG["batch_size"])
    test_dataloader = get_dataloader(test_dataset, CONFIG["batch_size"])

    model = MyResNET(args.freeze_params).to(DEVICE)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), CONFIG["lr"])
    criterion = nn.CrossEntropyLoss()

    wandb.init(project="compmed", group='ResNET')
    wandb.watch(model, log="all", log_freq=10)

    os.makedirs(args.out_dir, exist_ok=True)

    model_name= f'ResNET_lr{CONFIG["lr"]}_img{CONFIG["img_size"][0]}_b{CONFIG["batch_size"]}'
    best_model_path = os.path.join(args.out_dir, f"{model_name}.pth")

    trainer = Trainer(model, optimizer, criterion, DEVICE)
    trainer.train(CONFIG["n_epochs"], train_dataloader, val_dataloader, model_path=best_model_path)

    print("Done!\n")

    trainer.final_evaluate(test_dataloader)


if __name__ == "__main__":
    main()


