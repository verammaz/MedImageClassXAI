import os
import argparse
import torch
from torch import nn
from torch import optim
import wandb
from torchsummary import summary

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


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            nn.AdaptiveAvgPool2d(1), 
        )


        self.linear_stack = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_labels))

        self.name = 'cnn'


    def forward(self, x):
        x = x.to(next(self.parameters()).device)
        x = self.conv_stack(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.linear_stack(x)
        return x



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', required=True, help='path to datasets')
    parser.add_argument('-v', action='store_true', help='verbose mode')
    parser.add_argument('-out_dir', default='out', help='path where to save trained model parameters')
    parser.add_argument('-batch_size', default=32)
    parser.add_argument('-n_epochs', default=8)
    parser.add_argument('-lr', default=0.001)
    parser.add_argument('-w_decay', default=1e-5)
    parser.add_argument('-img_size', default=256)
    parser.add_argument('-img_channels', default=1)
    parser.add_argument('-img_norm', action='store_true')
    parser.add_argument('-pretrained', help='path to pretrained model')
    parser.add_argument('-wandb_project', default = 'ChestXRayClass')

    args = parser.parse_args()

    CONFIG = {'batch_size': int(args.batch_size),
              'n_epochs': int(args.n_epochs),
              'lr': float(args.lr),
              'w_decay': float(args.w_decay),
              'img_size': (int(args.img_size), int(args.img_size)),
              'img_norm': args.img_norm}

    assert(num_labels==2) # model applicable for binary classification


    train_transforms = get_train_transform(CONFIG['img_size'], CONFIG['img_norm'])

    common_transforms = get_common_transform(CONFIG['img_size'])

    train_dataset = get_dataset(args.data_dir, 'train', train_transforms)
    val_dataset = get_dataset(args.data_dir, 'val', common_transforms)
    test_dataset = get_dataset(args.data_dir, 'test', common_transforms)

    train_dataloader = get_dataloader(train_dataset, CONFIG["batch_size"], train=True)
    val_dataloader = get_dataloader(val_dataset, CONFIG["batch_size"])
    test_dataloader = get_dataloader(test_dataset, CONFIG["batch_size"])

    if args.v:
        print("Number of classes    : ", len(train_dataset.classes))
        print("Shape of image       : ", train_dataset[0][0].shape)
        print("Train batches        : ", train_dataloader.__len__())
        print("No. of train images  : ", train_dataset.__len__())
        print("No. of valid images  : ", val_dataset.__len__())
        print("No. of test images   : ", test_dataset.__len__())
        assert(train_dataset.class_to_idx == val_dataset.class_to_idx)
        print("Labels               : ", train_dataset.class_to_idx)

    os.makedirs(args.out_dir, exist_ok=True)

    model_name= f'SimpleCNN_lr{CONFIG["lr"]}_img{CONFIG["img_size"][0]}_b{CONFIG["batch_size"]}'
    if args.pretrained is not None: model_name += "_finetune" 
    best_model_path = os.path.join(args.out_dir, f"{model_name}.pth")


    model = SimpleCNN().to(DEVICE) 

    if args.v:
        summary(model, (1, CONFIG["img_size"][0], CONFIG["img_size"][1]))

    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    if args.pretrained is not None:
        print("Loading pretrained model...\n")
        model.load_state_dict(torch.load(args.pretrained, map_location=DEVICE, weights_only=True)) 
 
    else:
        model.apply(init_weights)

    wandb.init(project=args.wandb_project, group='SimpleCNN')
    wandb.watch(model, log="all", log_freq=10)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG['w_decay'])

    trainer = Trainer(model, optimizer, criterion, DEVICE)
    
    trainer.train(CONFIG["n_epochs"], train_dataloader, val_dataloader, model_path=best_model_path)

    print("Done!\n")

    trainer.final_evaluate(test_dataloader)

if __name__ == "__main__":
    main()

