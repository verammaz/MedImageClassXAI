import os
import json
import argparse
import torch
from tqdm import tqdm
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as T
import wandb
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np

wandb.login()

# Global Variables
img_size = (256, 256) # assuming square shape
img_shape = (1, 256, 256) # single channel (BW images)
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
CONFIG = {"batch_size": 64,
        "n_epochs": 8,
        "learning_rate": 0.001,
        "momentum": 0.9,
        "image_size": (256, 256),
        "image_normalize": False,
        }


class SimpleCNN(nn.Module):
    def __init__(self, image_size, num_labels):
        super(SimpleCNN, self).__init__()
        self.image_size = image_size  # Tuple (H, W)
        self.num_labels = num_labels

        self.layer_stack = nn.Sequential(
            nn.Conv2d(1, 128, (4, 4), stride=4, padding=0),  # Output: (128, H/4, W/4)
            nn.LayerNorm([128, self.image_size[0] // 4, self.image_size[1] // 4]),
            nn.Conv2d(128, 128, 7, stride=1, padding=3),     # Output: (128, H/4, W/4)
            nn.LayerNorm([128, self.image_size[0] // 4, self.image_size[1] // 4]),
            nn.Conv2d(128, 256, 1),                         # Output: (256, H/4, W/4)
            nn.GELU(),                                      # GELU activation
            nn.Conv2d(256, 128, 1),                         # Output: (128, H/4, W/4)
            nn.AvgPool2d(2, stride=2),                      # Output: (128, H/8, W/8)
            nn.Flatten(),                                   # Flatten: 128 * (H/8) * (W/8)
        )

        # Compute the flattened size dynamically
        h, w = self.image_size
        flattened_dim = 128 * (h // 8) * (w // 8)

        self.fc = nn.Linear(flattened_dim, self.num_labels)  # Fully connected layer

    def forward(self, x):
        # Ensure input tensor is on the same device as the model
        x = x.to(next(self.parameters()).device)
        x = self.layer_stack(x)
        logits = self.fc(x)
        return logits


def train_one_epoch(model, dataloader, optimizer, criterion, t):
    model.train()

    batch_size = dataloader.batch_size
    size = len(dataloader.dataset)

    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train')

    num_correct = 0.0
    total_loss = 0.0

    # Iterate all the training data and pass them into the model
    for idx, (images, labels) in enumerate(dataloader):

        optimizer.zero_grad() # zero all the gradients of the variable

        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # Forward propagation
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward propagation
        loss.backward()
        # Gradient descent
        optimizer.step()

        loss = float(loss.item())

        # Update no. of correct image predictions & loss as we iterate
        num_correct += int((torch.argmax(outputs, dim=1)==labels).sum())
        total_loss += loss

        # tqdm lets you add some details so you can monitor training as you train.
        batch_bar.set_postfix(
            acc = "{:.04f}%".format(num_correct/(batch_size*(idx+1))*100),
            loss = "{:.04f}".format(float(total_loss/(idx+1))),
        )

        batch_bar.update()

        wandb.log({"n_examples": (idx+1)*batch_size + size * t, "train_loss": loss})

    batch_bar.close()

 


def evaluate(model, dataloader, dataname, criterion):
    size = len(dataloader.dataset)
    batch_size = dataloader.batch_size

    model.eval()
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc=f'Evaluate on {dataname} dataset')

    avg_loss, num_correct = 0, 0

    TP, TN, FP, FN = 0, 0, 0, 0

    with torch.no_grad():
        for idx, (images, labels) in enumerate(dataloader):

            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            preds = torch.argmax(outputs, dim=1)

            num_correct += int((preds==labels).sum())
            avg_loss += float(loss.item())

            #FILL IN for sensitivity and specificity
            for p, l in zip(preds, labels):
                if p == 1 and l == 1:
                    TP += 1
                elif p == 0 and l == 0:
                    TN += 1
                elif p == 1 and l == 0:
                    FP += 1
                elif p == 0 and l == 1:
                    FN += 1

            batch_bar.set_postfix(
                acc= "{:.04f}%".format(num_correct/(batch_size*(idx+1))*100),
                loss= "{:.04f}".format(float(avg_loss/(idx+1)))
            )

            batch_bar.update()

    batch_bar.close()
    acc = num_correct/size
    total_loss = avg_loss/size

    return acc, total_loss, TP, TN, FP, FN




def get_image_transforms(config):
    transforms = [T.ToTensor(),
                T.Resize(min(config["image_size"][0], config["image_size"][1]), antialias=True),
                T.CenterCrop(config["image_size"])]

    transforms.append(T.Grayscale(num_output_channels=1)) # images are BW

    if config["image_normalize"]:
        transforms.append(T.Normalize(mean=[0.454], std=[0.282])) 

    return T.Compose(transforms)

def get_dataset(data_dir, type, image_transforms):
    dir = os.path.join(data_dir, type)

    # Hardcode label names --> TODO: make general by extracting from directory names
    class_to_label = {"NORMAL": 0, "PNEUMONIA": 1}

    dataset = datasets.ImageFolder(dir, transform=image_transforms)

    dataset.class_to_idx = class_to_label

    return dataset

def get_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', required=True, help='path to datasets')
    parser.add_argument('-config', help='path to config file for model')
    parser.add_argument('-v', action='store_true', help='verbose mode')
    parser.add_argument('-out_dir')

    args = parser.parse_args()

    image_transforms = get_image_transforms(CONFIG)

    train_dataset = get_dataset(args.data_dir, 'train', image_transforms)
    val_dataset = get_dataset(args.data_dir, 'val', image_transforms)
    test_dataset = get_dataset(args.data_dir, 'test', image_transforms)

    train_dataloader = get_dataloader(train_dataset, CONFIG["batch_size"])
    val_dataloader = get_dataloader(val_dataset, CONFIG["batch_size"])
    test_dataloader = get_dataloader(test_dataset, CONFIG["batch_size"])

    if args.v:
        print("Number of classes    : ", len(train_dataset.classes))
        print("Shape of image       : ", train_dataset[0][0].shape)
        print("Train batches        : ", train_dataloader.__len__())
        print("No. of train images  : ", train_dataset.__len__())
        print("No. of valid images  : ", val_dataset.__len__())
        assert(train_dataset.class_to_idx == val_dataset.class_to_idx)
        print("Labels               : ", train_dataset.class_to_idx)
        print("Image stats (mean, std): ", train_dataset[0][0].mean(), train_dataset[0][0].std())

    model = SimpleCNN(img_size, num_labels).to(DEVICE)

    if args.v:
        summary(model, img_shape)

    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    model.apply(init_weights)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    run = wandb.init(project="compmed")


    for t in range(CONFIG["n_epochs"]):
        print(f"\nEpoch {t+1}\n----------------------")
        train_one_epoch(model, train_dataloader, optimizer, criterion, t)
        train_loss, train_acc, *_ = evaluate(model, train_dataloader, 'Train', criterion)
        val_loss, val_acc, *_ = evaluate(model, val_dataloader, "Validation", criterion)
        test_loss, test_acc, *_ = evaluate(model, test_dataloader, "Test", criterion)
        wandb.log({"epoch": t, "train_loss_": train_loss, "val_loss_": val_loss, "train_acc_": train_acc, "val_acc_": val_acc, "test_loss_": test_loss, "test_acc_": test_acc})

    print("Done!")

    test_loss, test_acc, TP, TN, FP, FN = evaluate(model, test_dataloader, "Test", criterion)
    wandb.log({"Sensitivity": TP / (TP + FN), "Specificity": TN / (TN + FP)})

    # Save the model
    torch.save(model.state_dict(), os.path.join(args.out_dir, "model.pth"))
    print("Saved PyTorch Model State to model.pth")

if __name__ == "__main__":
    main()

