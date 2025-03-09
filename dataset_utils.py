import os
import random
import shutil
from math import ceil
import sys
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm 
import json
import kagglehub

def fetch_data_kaggle(data_dest):
    return 


def reorganize_dataset_kaggle(original_dataset, new_dataset, split_ratios=(0.8, 0.1, 0.1)):

    assert sum(split_ratios) == 1.0
    
    class_names = os.listdir(os.path.join(original_dataset, 'train'))  # assuming all sets have the same classes

    for class_name in class_names:
        print(f"processing {class_name} files...")
        
        # combine all data into a single list
        all_data = []
        if class_name.startswith('.'): continue
        
        for folder in ['train', 'val', 'test']:
            class_folder = os.path.join(original_dataset, folder, class_name)
            if os.path.exists(class_folder):
                images = [os.path.join(class_folder, img) for img in os.listdir(class_folder)]
                all_data.extend(images)

        # shuffle combined data
        random.shuffle(all_data)

        # calculate split sizes
        total_images = len(all_data)
        train_size = ceil(split_ratios[0] * total_images)
        val_size = ceil(split_ratios[1] * total_images)

        train_data = all_data[:train_size]
        val_data = all_data[train_size:train_size + val_size]
        test_data = all_data[train_size + val_size:]

        # create new folder structure
        for split, split_data in zip(['train', 'val', 'test'], [train_data, val_data, test_data]):
            print(f"creating {split} folder...")
            for img_path in split_data:
                split_class_dir = os.path.join(new_dataset, split, class_name)
                os.makedirs(split_class_dir, exist_ok=True)
                shutil.copy(img_path, os.path.join(split_class_dir, os.path.basename(img_path)))
        print('\n')
 



def organize_dataset_nih(data_entry_file, data_path, split_ratios=(0.8, 0.1, 0.1)):

    data_entry_df = pd.read_csv(data_entry_file)
    pneumonia_imgs, normal_imgs = [], []

    for _, row in data_entry_df.iterrows():
        if 'Pneumonia' in row['Finding Labels']:
            pneumonia_imgs.append(os.path.join(data_path, 'images', row['Image Index']))
        elif 'No Finding' in row['Finding Labels']:
            normal_imgs.append(os.path.join(data_path, 'images', row['Image Index']))
    
    for class_name, img_list in [('PNEUMONIA', pneumonia_imgs), ('NORMAL', normal_imgs)]:
        print(f"processing {class_name} files...")
        total_images = len(img_list)
        train_size = ceil(split_ratios[0] * total_images)
        val_size = ceil(split_ratios[1] * total_images)

        train_data = img_list[:train_size]
        val_data = img_list[train_size:train_size + val_size]
        test_data = img_list[train_size + val_size:]

        # create new folder structure
        for split, split_data in zip(['train', 'val', 'test'], [train_data, val_data, test_data]):
            print(f"creating {split} folder...")
            for img_path in split_data:
                split_class_dir = os.path.join(data_path, split, class_name)
                os.makedirs(split_class_dir, exist_ok=True)
                shutil.copy(img_path, os.path.join(split_class_dir, os.path.basename(img_path)))

        print('\n')



def get_dataset_counts(dataset_path):

    class_counts = {"train": {"NORMAL": 0, "PNEUMONIA": 0}, 
                    "test": {"NORMAL": 0, "PNEUMONIA": 0}, 
                    "val": {"NORMAL": 0, "PNEUMONIA": 0}}

    for split in class_counts.keys():
        folder_path = os.path.join(dataset_path, split)
        
        if not os.path.exists(folder_path):
            print(f"Warning: Folder '{folder_path}' does not exist.")
            continue
        
        for class_name in class_counts[split].keys():
            class_folder_path = os.path.join(folder_path, class_name)
            
            if os.path.exists(class_folder_path):
                class_counts[split][class_name] = len([
                    file for file in os.listdir(class_folder_path) 
                    if os.path.isfile(os.path.join(class_folder_path, file))
                ])
            else:
                print(f"Warning: Folder '{class_folder_path}' does not exist.")

    totals = {"NORMAL": 0, "PNEUMONIA": 0, "Total": 0}
    for folder, counts in class_counts.items():
        for class_name, count in counts.items():
            totals[class_name] += count
            totals["Total"] += count

    df = pd.DataFrame(class_counts).T
    df["Total"] = df["NORMAL"] + df["PNEUMONIA"]
    totals_df = pd.DataFrame([totals], index=["Total"])

    result_df = pd.concat([df, totals_df])

    # save results
    result_df.to_csv(os.path.join(dataset_path, 'dataset_counts.txt'), sep='\t')



def get_dataset_stats(dataset_path):

    transform = transforms.Compose([transforms.ToTensor()])

    sum_pixel_values = 0.0
    sum_squared_pixel_values = 0.0
    total_pixels = 0

    for folder in ["train", "test", "val"]:
        folder_path = os.path.join(dataset_path, folder)
        
        for class_name in ["NORMAL", "PNEUMONIA"]:
            class_folder_path = os.path.join(folder_path, class_name)
            
            if os.path.exists(class_folder_path):
                for image_name in tqdm(os.listdir(class_folder_path), desc=f"Processing {folder}/{class_name}"):
                    image_path = os.path.join(class_folder_path, image_name)
                    
                    # open image
                    with Image.open(image_path) as img:
                        # convert to tensor
                        img_tensor = transform(img)
                        
                        # accumulate pixel values and squared pixel values
                        sum_pixel_values += img_tensor.sum()
                        sum_squared_pixel_values += (img_tensor ** 2).sum()
                        total_pixels += img_tensor.numel()  

    # compute mean and std
    mean = sum_pixel_values / total_pixels
    std = (sum_squared_pixel_values / total_pixels - mean ** 2).sqrt()

    # save results to json
    with open(os.path.join(dataset_path, "dataset_stats.json"), "w") as f:
        json.dump({"mean": mean, "std": std}, f)

