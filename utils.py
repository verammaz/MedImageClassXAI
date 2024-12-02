import os
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets
import torchvision.transforms as T


def get_train_transform(img_size, norm=True):
    transforms = [T.RandomHorizontalFlip(),
                    T.RandomRotation(15),
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    T.Resize(img_size),
                    T.Grayscale(num_output_channels=1),  # Convert to grayscale (1 channel)
                    T.ToTensor()]

    if norm:
        transforms.append(T.Normalize(mean=[0.4881], std=[0.2442])) # stats of kaggle chest_xray data: paultimothymooney/chest-xray-pneumonia 

    return T.Compose(transforms)


def get_common_transform(img_size):
    transforms = [T.Grayscale(num_output_channels=1),  
                T.Resize(img_size),
                T.ToTensor(),
                T.Normalize(mean=[0.4881], std=[0.2442])]
    
    return T.Compose(transforms)


def get_dataset(data_dir, type, image_transforms):
    dir = os.path.join(data_dir, type)

    # Hardcode label names 
    class_to_label = {"NORMAL": 0, "PNEUMONIA": 1}

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