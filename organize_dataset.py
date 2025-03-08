import os
import random
import shutil
from math import ceil
import sys

def organize_dataset(original_dataset, new_dataset, split_ratios=(0.8, 0.1, 0.1)):
    """
    Combine, shuffle, and split dataset into train, val, and test sets.

    :param original_dataset: Path to the original dataset folder
    :param new_dataset: Path to the new dataset folder
    :param split_ratios: Tuple indicating the train, val, test split ratios
    """
    assert sum(split_ratios) == 1.0
    
    # Combine all data into a single list
    all_data = []
    class_names = os.listdir(os.path.join(original_dataset, 'train'))  # Assuming all sets have the same classes

    for class_name in class_names:
        if class_name.startswith('.'): continue
        for folder in ['train', 'val', 'test']:
            class_folder = os.path.join(original_dataset, folder, class_name)
            if os.path.exists(class_folder):
                images = [(os.path.join(class_folder, img), class_name) for img in os.listdir(class_folder)]
                all_data.extend(images)

    # Shuffle the combined data
    random.shuffle(all_data)

    # Calculate split sizes
    total_images = len(all_data)
    train_size = ceil(split_ratios[0] * total_images)
    val_size = ceil(split_ratios[1] * total_images)

    train_data = all_data[:train_size]
    val_data = all_data[train_size:train_size + val_size]
    test_data = all_data[train_size + val_size:]

    # Create new folder structure
    for split, split_data in zip(['train', 'val', 'test'], [train_data, val_data, test_data]):
        for img_path, class_name in split_data:
            split_class_dir = os.path.join(new_dataset, split, class_name)
            os.makedirs(split_class_dir, exist_ok=True)
            shutil.copy(img_path, os.path.join(split_class_dir, os.path.basename(img_path)))

    print(f"Dataset organized successfully into {new_dataset}")
    print(f"Train: {len(train_data)} images, Val: {len(val_data)} images, Test: {len(test_data)} images")


if __name__ == "main":
    organize_dataset(sys.argv[1], sys.argv[2])