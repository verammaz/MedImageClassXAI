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
    def __init__(self, image_channels, image_size, num_labels):
        super(SimpleCNN, self).__init__()

        self.layer_stack = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (H/2, W/2)

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (H/4, W/4)

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # Global average pooling to (256, 1, 1)
        )
        self.fc = nn.Linear(256, num_labels)

    def forward(self, x):
        x = x.to(next(self.parameters()).device)
        x = self.layer_stack(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x



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




def get_image_transforms(config):
    transforms = [T.ToTensor(),
                T.Resize(min(config["img_size"][0], config["img_size"][1]), antialias=True),
                T.CenterCrop(config["img_size"])]

    transforms.append(T.Grayscale(num_output_channels=config["img_channels"])) 
    transforms.append(T.RandomHorizontalFlip(p=0.5))
    transforms.append(T.RandomRotation(10))
    transforms.append(T.ColorJitter(brightness=0.2, contrast=0.2))

    if config["img_norm"]:
        # TODO: make more general --> calculate mean/std dynamically
        transforms.append(T.Normalize(mean=[0.5519], std=[0.2131])) # stats of kaggle chest_xray data: paultimothymooney/chest-xray-pneumonia 

    return T.Compose(transforms)


def get_dataset(data_dir, type, image_transforms):
    dir = os.path.join(data_dir, type)

    # Hardcode label names --> TODO: make general by extracting from directory names
    class_to_label = {"NORMAL": 0, "PNEUMONIA": 1}
    assert(len(class_to_label) == 2)
    dataset = datasets.ImageFolder(dir, transform=image_transforms)
    dataset.class_to_idx = class_to_label

    return dataset


def get_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', required=True, help='path to datasets')
    parser.add_argument('-v', action='store_true', help='verbose mode')
    parser.add_argument('-out_dir', default='out', help='path where to save trained model parameters')
    parser.add_argument('-batch_size', default=64)
    parser.add_argument('-n_epochs', default=8)
    parser.add_argument('-lr', default=0.001)
    parser.add_argument('-img_size', default=(256, 256))
    parser.add_argument('-img_channels', default=1)
    parser.add_argument('-img_norm', action='store_true')

    args = parser.parse_args()

    CONFIG = {'batch_size': int(args.batch_size),
              'n_epochs': int(args.n_epochs),
              'lr': args.lr,
              'img_size': args.img_size,
              'img_norm': args.img_norm,
              "img_channels": args.img_channels}

    assert(num_labels==2) # model applicable for binary classification

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

    model = SimpleCNN(CONFIG["img_channels"], CONFIG["img_size"], num_labels).to(DEVICE)

    if args.v:
        summary(model, (CONFIG["img_channels"], CONFIG["img_size"][0], CONFIG["img_size"][1]))

    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    model.apply(init_weights)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])

    wandb.init(project="compmed")

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
        
        test_loss, test_acc, sens, spec = evaluate(model, test_dataloader, "Test", criterion)
        wandb.log({"test_specificity": spec, "test_sensitivity": sens})
        print(f'Test Loss: {test_loss}, Test Accuracy: {test_acc}')
        
        wandb.log({"epoch": t, "train_loss_": train_loss, 
                                "val_loss_": val_loss, 
                                "train_acc_": train_acc, 
                                "val_acc_": val_acc, 
                                "test_loss_": test_loss, 
                                "test_acc_": test_acc})

        # Save the best model based on validation accuracy or loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.out_dir, "SimpleCNN_model.pth"))
            print(f"Best model saved with Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

    print("Done!\n")


    print("Final Model Performance:\n-------------------")
    test_loss, test_acc, sensitivity, specificity, cm = evaluate(model, test_dataloader, "Test", criterion, confusion_matrix=True)
    print(f'Loss: {test_loss}, Accuracy: {test_acc}')
    print(f'Confusion Matrix\n: {cm}')
    print(f'Sensitivity: {sensitivity}, Specificity: {specificity}')
    wandb.log({"Sensitivity": sensitivity, "Specificity": specificity})


if __name__ == "__main__":
    main()

