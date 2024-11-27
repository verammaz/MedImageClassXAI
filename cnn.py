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

#wandb.login()

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
CONFIG = {"batch_size": 8,
        "epoch": 3,
        "learning_rate": 0.01,
        "momentum": 0.9,
        "image_size": (256, 256),
        "image_normalize": True,
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

def train(model, dataloader, optimizer, criterion):
    model.train()

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

        # Update no. of correct image predictions & loss as we iterate
        num_correct += int((torch.argmax(outputs, dim=1)==labels).sum())
        total_loss += float(loss.item())

        # tqdm lets you add some details so you can monitor training as you train.
        batch_bar.set_postfix(
            acc = "{:.04f}%".format(num_correct/(CONFIG["batch_size"]*(idx+1))*100),
            loss = "{:.04f}".format(float(total_loss/(idx+1))),
        )

        batch_bar.update()

    batch_bar.close()

    # Calculate the total accuracy and loss for this epoch
    acc = num_correct/(CONFIG["batch_size"]*len(dataloader))*100
    total_loss = float(total_loss/len(dataloader))

    return acc, total_loss


def validate(model, dataloader, criterion):
    model.eval()
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc='Validate')

    num_correct = 0.0
    total_loss = 0.0

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for idx, (images, labels) in enumerate(dataloader):

        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # Get model outputs
        # For validation, we use the inference mode
        with torch.inference_mode():
            outputs = model(images)
            loss = criterion(outputs, labels)

        preds = torch.argmax(outputs, dim=1)

        num_correct += int((preds==labels).sum())
        total_loss += float(loss.item())

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
            acc= "{:.04f}%".format(num_correct/(CONFIG["batch_size"]*(idx+1))*100),
            loss= "{:.04f}".format(float(total_loss/(idx+1)))
        )

        batch_bar.update()

    batch_bar.close()
    acc = num_correct/(CONFIG["batch_size"]*len(dataloader))*100
    total_loss = float(total_loss/len(dataloader))

    return acc, total_loss, TP, TN, FP, FN

def train_(model, train_loader, valid_loader, optimizer, criterion, config):
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    # Iterate over the number epochs you specified in your config dictionary.
    for epoch in range(config["epoch"]):
        # Train your model
        train_acc, train_loss = train(model, train_loader, optimizer, criterion)

        print("\nEpoch {}/{}: \nTrain Acc {:.04f}%\t Train Loss {:.04f}\t Learning Rate {:.04f}".format(
            epoch + 1,
            config["epoch"],
            train_acc,
            train_loss,
            config["learning_rate"]))

        # Validate your model
        val_acc, val_loss, TP, TN, FP, FN = validate(model, valid_loader, criterion)

        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)

        print("Val Acc {:.04f}%\t Val Loss {:.04f}".format(val_acc, val_loss))
        print("Sensitivity {:.04f}%\t Specificity {:.04f}%".format(sensitivity * 100, specificity * 100))

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        return train_loss_list, val_loss_list, train_acc_list, val_acc_list


def plot_training(config, train_loss_list, val_loss_list, train_acc_list, val_acc_list):
    epochs = [i for i in range(1, config["epoch"] + 1)]

    plt.figure
    plt.plot(epochs, train_loss_list, label='Training Loss')
    plt.plot(epochs, val_loss_list, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    plt.figure
    plt.plot(epochs, train_acc_list, label='Training Accuracy')
    plt.plot(epochs, val_acc_list, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()


def test(model, dataloader):
    model.eval()
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc='Test')

    pred_labels = []
    true_labels = []

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for _, (images, labels) in enumerate(dataloader):
        images = images.to(DEVICE)

        with torch.inference_mode():
            outputs = model(images)

        # Get predictions and true labels
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy().tolist()
        labels = labels.detach().cpu().numpy().tolist()

        # Extend lists to track all predictions and labels
        pred_labels.extend(preds)
        true_labels.extend(labels)

        # Update TP, TN, FP, FN counts
        for p, l in zip(preds, labels):
          if p == 1 and l == 1:
              TP += 1
          elif p == 0 and l == 0:
            TN += 1
          elif p == 1 and l == 0:
            FP += 1
          elif p == 0 and l == 1:
            FN += 1

        batch_bar.update()

    batch_bar.close()

    # Calculate sensitivity and specificity
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    print("Sensitivity {:.04f}%\t Specificity {:.04f}%".format(sensitivity * 100, specificity * 100))

    return pred_labels, true_labels



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

    args = parser.parse_args()
    config = args.config

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

    model = SimpleCNN(img_size, num_labels).to(DEVICE)

    if args.v:
        summary(model, img_shape)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=CONFIG["learning_rate"], momentum=CONFIG["momentum"])

    train_loss_list, val_loss_list, train_acc_list, val_acc_list = train_(model, train_dataloader, val_dataloader, optimizer, criterion, CONFIG)
    plot_training(CONFIG, train_loss_list, val_loss_list, train_acc_list, val_acc_list)
    
    pred_labels, true_labels = test(model, test_dataloader)

    # TODO: calculate the accuracy score
    test_acc = (np.array(pred_labels) == np.array(true_labels)).mean() * 100
    print("Test Acc {:.04f}%".format(test_acc))

if __name__ == "__main__":
    main()

