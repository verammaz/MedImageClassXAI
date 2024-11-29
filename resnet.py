import torch
import torch.nn as nn
from torchvision.models import resnet18
import wandb
import os
import json
import argparse
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets
import torchvision.transforms as T
import wandb
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
from cnn import SimpleCNN

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

class MyResNET():
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
    

def get_image_transforms(config):
    transforms = [T.RandomHorizontalFlip(),
                T.RandomRotation(15),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.Resize(config["img_size"]),
                T.Grayscale(num_output_channels=1),  # Convert to grayscale (1 channel)
                T.ToTensor()]

    if config["img_norm"]:
        # TODO: make more general --> calculate mean/std dynamically
        transforms.append(T.Normalize(mean=[0.5519], std=[0.2131])) # stats of kaggle chest_xray data: paultimothymooney/chest-xray-pneumonia 


    transforms_common = [T.Grayscale(num_output_channels=1),  
            T.Resize(config["img_size"]),
            T.ToTensor(),
            T.Normalize(mean=(0.5,), std=(0.5,))]
    

    return T.Compose(transforms), T.Compose(transforms_common)


def get_dataset(data_dir, type, image_transforms):
    dir = os.path.join(data_dir, type)

    # Hardcode label names --> TODO: make general by extracting from directory names
    class_to_label = {"NORMAL": 0, "PNEUMONIA": 1}
    assert(len(class_to_label) == 2)
    dataset = datasets.ImageFolder(dir, transform=image_transforms)
    dataset.class_to_idx = class_to_label

    return dataset


def get_dataloader(dataset, batch_size, train=False):
    if train:
        # Weighted Random Sampling for dealing with Imbalanced Dataset
        class_freq = torch.as_tensor(dataset.targets).bincount()
        weight = 1 / class_freq
        samples_weight = weight[dataset.targets]
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=2)
    
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)


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

        wandb.log({"n_examples": (idx+1)*batch_size + size * t, "train_loss": loss, "train_accuracy": num_correct/(batch_size*(idx+1))*100})

    batch_bar.close()

 


def evaluate(model, dataloader, dataname, criterion, confusion_matrix=False):
    batch_size = dataloader.batch_size

    model.eval()
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc=f'Evaluate on {dataname} dataset')

    total_loss, num_correct = 0, 0

    TP, TN, FP, FN = 0, 0, 0, 0

    with torch.no_grad():
        for idx, (images, labels) in enumerate(dataloader):

            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            preds = torch.argmax(outputs, dim=1)

            num_correct += int((preds==labels).sum())
            total_loss += float(loss.item())

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
                loss= "{:.04f}".format(float(total_loss/(idx+1)))
            )

            batch_bar.update()

    batch_bar.close()

    acc = num_correct/(batch_size*(idx+1))*100
    avg_loss = total_loss/(idx+1)

    sensitivity, specificity =  TP / (TP + FN), TN / (TN + FP)

    if not confusion_matrix:
        return avg_loss, acc, sensitivity, specificity
    
    else:
        cm = np.array([[TP, FP], [FN, TN]])
        return avg_loss, acc, sensitivity, specificity, cm
    


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

    args = parser.parse_args()

    CONFIG = {'batch_size': int(args.batch_size),
              'n_epochs': int(args.n_epochs),
              'lr': args.lr,
              'img_size': (int(args.img_size), int(args.img_size)),
              'img_norm': args.img_norm}

    assert(num_labels==2) # model applicable for binary classification

    
    img_transforms, _ = get_image_transforms(CONFIG)

    train_dataset = get_dataset(args.data_dir, 'train', img_transforms)
    val_dataset = get_dataset(args.data_dir, 'val', img_transforms)
    test_dataset = get_dataset(args.data_dir, 'test', img_transforms)

    train_dataloader = get_dataloader(train_dataset, CONFIG["batch_size"], train=True)
    val_dataloader = get_dataloader(val_dataset, CONFIG["batch_size"])
    test_dataloader = get_dataloader(test_dataset, CONFIG["batch_size"])

    model = MyResNET(args.freeze_params).to(DEVICE)

    print(f"Device: {DEVICE}")

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), CONFIG["lr"])
    criterion = nn.CrossEntropyLoss()


    wandb.init(project="compmed", group='ResNET')
    wandb.watch(model, log="all", log_freq=10)

    os.makedirs(args.out_dir, exist_ok=True)

    best_val_loss = np.inf
    best_val_acc = 0.0

    for t in range(CONFIG["n_epochs"]):
        print(f"\nEpoch {t+1}\n----------------------")
        train_one_epoch(model, train_dataloader, optimizer, criterion, t)
        
        train_loss, train_acc, sens, spec = evaluate(model, train_dataloader, 'Train', criterion)
        wandb.log({"train_specificity": spec, "train_sensitivity": sens})
        print(f'Train Loss: {train_loss}, Train Accuracy: {train_acc}')
        
        val_loss, val_acc, sens, spec = evaluate(model, val_dataloader, "Validation", criterion)
        wandb.log({"val_specificity": spec, "val_sensitivity": sens})
        print(f'Val Loss: {val_loss}, Val Accuracy: {val_acc}')
        
        #test_loss, test_acc, sens, spec = evaluate(model, test_dataloader, "Test", criterion)
        #wandb.log({"test_specificity": spec, "test_sensitivity": sens})
        #print(f'Test Loss: {test_loss}, Test Accuracy: {test_acc}')
        
        wandb.log({"epoch": t, 
                   "train_loss_": train_loss, 
                   "val_loss_": val_loss, 
                   "train_acc_": train_acc, 
                   "val_acc_": val_acc, 
                   #"test_loss_": test_loss, 
                   #"test_acc_": test_acc
                   })


        # Save the best model based on validation accuracy or loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            model_name = f'ResNET_lr{CONFIG["lr"]}_img{CONFIG["img_size"][0]}_b{CONFIG["batch_size"]}'
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"{model_name}.pth"))
            print(f"Best model saved with Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")


    print("Done!\n")

    total_params = sum(
	param.numel() for param in model.parameters()
    )

    print(f"\nTotal parameters: {total_params}\n")

    print("Final Model Performance:\n-------------------")
    test_loss, test_acc, sensitivity, specificity, cm = evaluate(model, test_dataloader, "Test", criterion, confusion_matrix=True)
    print(f'Loss: {test_loss}, Accuracy: {test_acc}')
    print(f'Confusion Matrix\n: {cm}')
    print(f'Sensitivity: {sensitivity}, Specificity: {specificity}')
    wandb.log({"Sensitivity": sensitivity, "Specificity": specificity})


if __name__ == "__main__":
    main()


