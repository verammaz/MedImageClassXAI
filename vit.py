import os
from PIL import Image
import argparse
import torch
from tqdm import tqdm
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision import datasets
import torchvision.transforms as T
import wandb
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
from transformers import ViTForImageClassification, ViTImageProcessor


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
    

def get_image_transforms(config):
    transforms = [T.RandomHorizontalFlip(),
                T.RandomRotation(15),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.Resize((224, 224)),
                T.Grayscale(num_output_channels=3),  # Convert to grayscale RGB
                T.ToTensor()]

    if config["img_norm"]:
        transforms.append(T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])) # stats of kaggle chest_xray data: paultimothymooney/chest-xray-pneumonia 
    

    return T.Compose(transforms)


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
    
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)


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
    
        outputs = model(images)
        logits = outputs.logits
   
        loss = criterion(logits, labels)

        # Backward propagation
        loss.backward()
        # Gradient descent
        optimizer.step()

        loss = float(loss.item())

        # Update no. of correct image predictions & loss as we iterate
        num_correct += int((torch.argmax(logits, dim=1)==labels).sum())
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
            logits = outputs.logits
            loss = criterion(logits, labels)

            preds = torch.argmax(logits, dim=1)

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


    #img_transforms = get_image_transforms(CONFIG)

    #train_dataset = get_dataset(args.data_dir, 'train', img_transforms)
    #val_dataset = get_dataset(args.data_dir, 'val', img_transforms)
    #test_dataset = get_dataset(args.data_dir, 'test', img_transforms)


    train_dataset = PneumoniaDataset(os.path.join(args.data_dir, 'train'), processor, transform=transform)
    val_dataset = PneumoniaDataset(os.path.join(args.data_dir, 'val'), processor, transform=transform)
    test_dataset = PneumoniaDataset(os.path.join(args.data_dir, 'test'), processor, transform=transform)

    train_dataloader = get_dataloader(train_dataset, CONFIG["batch_size"], train=False)
    val_dataloader = get_dataloader(val_dataset, CONFIG["batch_size"])
    test_dataloader = get_dataloader(test_dataset, CONFIG["batch_size"])


    model = MyViT().to(DEVICE)

    print(f"Device: {DEVICE}")

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), CONFIG["lr"])
    criterion = nn.CrossEntropyLoss()


    wandb.init(project="compmed", group='ViT')
    wandb.watch(model, log="all", log_freq=10)

    os.makedirs(args.out_dir, exist_ok=True)

    best_val_loss = np.inf
    best_val_acc = 0.0

    model_name= f'ViT_lr{CONFIG["lr"]}_b{CONFIG["batch_size"]}'
    best_model_path = os.path.join(args.out_dir, f"{model_name}.pth")

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
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path}")


    print("Done!\n")

    total_params = sum(param.numel() for param in model.parameters())
    print(f"\nTotal parameters: {total_params}\n")

    # Reload the best model
    model.load_state_dict(torch.load(best_model_path))
    model.eval()


    print("Best Model Performance:\n-------------------")
    test_loss, test_acc, sensitivity, specificity, cm = evaluate(model, test_dataloader, "Test", criterion, confusion_matrix=True)
    print(f'Loss: {test_loss}, Accuracy: {test_acc}')
    print(f'Confusion Matrix\n: {cm}')
    print(f'Sensitivity: {sensitivity}, Specificity: {specificity}')
    wandb.log({"Sensitivity": sensitivity, "Specificity": specificity})


if __name__ == "__main__":
    main()


