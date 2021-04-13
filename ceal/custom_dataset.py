import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torchvision import transforms, models

import os
from PIL import Image
import numpy as np
import yaml

data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class CustomDataset(Dataset):
    def __init__(self, root_dir, labeled, class_label, transform=None):
        self.root_dir = os.path.expanduser(root_dir)
        self.class_label = class_label

        self.data = []
        self.labels = []
        self.class_label = class_label
        self._classes = len(self.class_label)
        self.transform = transform

        if labeled:
            for i, label in enumerate(class_label):
                label_dir = os.path.join(self.root_dir, label)
                files = os.listdir(label_dir)
                for file in files:
                    self.data.append(os.path.join(label_dir, file))
                    self.labels.append(i)
        else:
            files = os.listdir(self.root_dir)
            for file in files:
                self.data.append(os.path.join(root_dir, file))
                self.labels.append(-1)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx], self.labels[idx]
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)
        
        sample = {'image': img, 'label': label, 'filename': img_path}
        return sample
    
    def __len__(self):
        return len(self.data)

def make_labeled_dataloader(configuration_path):
    
    with open(configuration_path, 'r') as cfg_path:
        cfg = yaml.safe_load(cfg_path)
    
    labeled_dataset_path = cfg['config']['labeled_data_path']
    labeled_category = cfg['config']['labeled_category']
    labeled_dataset = CustomDataset(labeled_dataset_path, True, 
                                    labeled_category, data_transform)

    random_seed = 42
    validation_ratio = 0.2
    dataset_size = len(labeled_dataset)
    dataset_indices = list(range(dataset_size))

    np.random.seed(random_seed)
    np.random.shuffle(dataset_indices)

    val_split_index = int(np.floor(dataset_size * validation_ratio))
    train_idx, val_idx = dataset_indices[val_split_index:], dataset_indices[:val_split_index]
    
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    batch_size = cfg['config']['batch_size']
    dl = DataLoader(labeled_dataset, batch_size=batch_size,
                    sampler=train_sampler, num_workers=4)
    dtest = DataLoader(labeled_dataset, batch_size=batch_size,
                    sampler=val_sampler, num_workers=4)
        
    return dl, dtest

def make_unlabeled_dataloader(configuration_path):
    
    with open(configuration_path, 'r') as cfg_path:
        cfg = yaml.safe_load(cfg_path)
    
    batch_size = cfg['config']['batch_size']
    unlabeled_dataset_path = cfg['config']['unlabeled_data_path']
    unlabeled_dataset = CustomDataset(unlabeled_dataset_path, False,
                                    [], data_transform)
    
    du = DataLoader(unlabeled_dataset, batch_size=batch_size,
                    num_workers=4)
    
    return du