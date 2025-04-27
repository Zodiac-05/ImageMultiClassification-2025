# dataset.py
import os
import random
from collections import defaultdict
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from config import Config

def parse_filename(filename):
    name = os.path.splitext(filename)[0]
    if '_g' in name:
        exercise, rest = name.split('_g')
        return exercise.strip(), f'google_{rest}'
    else:
        parts = name.split('_')
        exercise = parts[0].strip()
        video_num = parts[1][:2] if len(parts[1]) > 5 else parts[1][0]
        return exercise, f'video_{video_num}'

class WorkoutDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

def create_dataloader():
    video_groups = defaultdict(list)
    class_to_idx = {}
    idx = 0

    for exercise in os.listdir(Config.data_dir):
        exercise_path = os.path.join(Config.data_dir, exercise)
        if os.path.isdir(exercise_path):
            class_to_idx[exercise] = idx
            idx += 1
            for img_file in os.listdir(exercise_path):
                if img_file.endswith('.jpg'):
                    img_path = os.path.join(exercise_path, img_file)
                    _, group_id = parse_filename(img_file)
                    video_groups[(exercise, group_id)].append((img_path, class_to_idx[exercise]))

    group_keys = list(video_groups.keys())
    random.seed(Config.random_seed)
    random.shuffle(group_keys)

    split_idx = int(len(group_keys) * (1 - Config.test_size))
    train_keys = group_keys[:split_idx]
    test_keys = group_keys[split_idx:]

    train_samples = []
    test_samples = []

    for key in train_keys:
        train_samples.extend(video_groups[key])
    for key in test_keys:
        test_samples.extend(video_groups[key])

    train_image_paths = [x[0] for x in train_samples]
    train_labels = [x[1] for x in train_samples]
    test_image_paths = [x[0] for x in test_samples]
    test_labels = [x[1] for x in test_samples]

    train_transform = transforms.Compose([
        transforms.Resize((Config.resize_x + 32, Config.resize_y + 32)),
        transforms.RandomCrop((Config.resize_x, Config.resize_y)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((Config.resize_x, Config.resize_y)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = WorkoutDataset(train_image_paths, train_labels, transform=train_transform)
    test_dataset = WorkoutDataset(test_image_paths, test_labels, transform=test_transform)

    return train_dataset, test_dataset, class_to_idx
