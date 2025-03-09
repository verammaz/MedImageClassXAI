import os
import json
import argparse
import random
import string
import torch
from torch import nn
from torch import optim
from torchsummary import summary

from train import Trainer
from utils import *


# Global Variables
lablel_to_str = {0: "normal", 1: "pneumonia"}
DEFAULTS = {'batch_size': 64,
              'n_epochs': 10,
              'lr': 0.001,
              'w_decay': 1e-5,
              'img_size': (256, 256),
              'img_norm': False}

# Get device for training
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


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
    parser.add_argument('-data_dir', required=True, help='path to train/val/test data folders')
    parser.add_argument('-v', action='store_true', help='verbose mode')
    parser.add_argument('-train_dir', required=True, help='path to training logs')
    parser.add_argument('-models_dir', required=True, help='path to where models saved')
    parser.add_argument('-config_file', required=False, help='json file with training hyperparameters')
    parser.add_argument('-pretrained', help='path to pretrained model')

    args = parser.parse_args()

    if not os.path.exists(args.train_dir): os.makedirs(args.train_dir)
    if not os.path.exists(args.models_dir): os.makedirs(args.models_dir)

    if args.config_file is not None:
        with open(args.config_file, 'r') as f:
            config = json.load(f)
    else:
        config = dict()

    # check hparams and fill missing with default values
    hparams = ['batch_size', 'n_epochs', 'lr', 'w_decay', 'img_size', 'img_norm']
    for hparam in hparams:
        if hparam not in config:
            print(f"hyperparameter {hparam} not specified, setting to default value: {hparam}={DEFAULTS[hparam]}")
            config[hparam] = DEFAULTS[hparam]

    
    # if normalizing images, get mean and std stats
    if config['img_norm']:

        if not os.path.isfile(os.path.join(args.data_dir, "dataset_stats.json")):
            from dataset_utils import get_dataset_stats
            get_dataset_stats(args.data_dir)
        
        with open(os.path.join(args.data_dir, "dataset_stats.json"), 'r') as f:
            stats = json.load(f)
        
        config['img_mean'] = stats['mean']
        config['img_std'] = stats['std']


    train_transforms = get_train_transform(config['img_size'], img_mean=config.get('img_mean'), img_std=config.get('img_std'))

    common_transforms = get_common_transform(config['img_size'])

    train_dataset = get_dataset(args.data_dir, 'train', train_transforms)
    val_dataset = get_dataset(args.data_dir, 'val', common_transforms)
    test_dataset = get_dataset(args.data_dir, 'test', common_transforms)

    train_dataloader = get_dataloader(train_dataset, config["batch_size"], train=True)
    val_dataloader = get_dataloader(val_dataset, config["batch_size"])
    test_dataloader = get_dataloader(test_dataset, config["batch_size"])

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

    model_name= f'SimpleCNN_lr{config["lr"]}_img{config["img_size"][0]}_b{config["batch_size"]}'
    if args.pretrained is not None: model_name += "_finetune" 
    best_model_path = os.path.join(args.models_dir, f"{model_name}.pth")


    model = SimpleCNN().to(DEVICE) 

    if args.v:
        summary(model, (1, config["img_size"][0], config["img_size"][1]))

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

    # generate run name
    run_name = ''
    chars = string.ascii_letters + string.digits
    while True:
        run_name = ''.join(random.choice(chars) for _ in range(10))
        if not os.path.exists(os.path.join(args.train_dir, run_name)):
            break
    
    # save hparams
    config['run'] = run_name
    config['model'] = model_name
    with open(os.path.join(args.train_dir, run_name, 'hparams.json'), 'w') as f:
        json.dumps(config, f)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config['w_decay'])

    trainer = Trainer(model, optimizer, criterion, DEVICE)
    run_path = os.path.join(args.train_dir, run_name)
    trainer.train(config["n_epochs"], train_dataloader, val_dataloader, run_path=run_path, model_path=best_model_path)

    print("Done!\n")

    trainer.final_evaluate(test_dataloader, run_path)

if __name__ == "__main__":
    main()

