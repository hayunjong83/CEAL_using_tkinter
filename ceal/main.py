import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, models

from custom_dataset import CustomDataset
import yaml
import os
import numpy as np

from ceal import ceal

def main():

    configuration_path = 'configuration.yml' 
    if not os.path.isfile(configuration_path):
        print('There is no configuration file. Please Check it!')
        return
    
    with open(configuration_path, 'r') as cfg_path:
        cfg = yaml.safe_load(cfg_path)
    
    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    labeled_dataset_path = cfg['config']['labeled_data_path']
    labeled_category = cfg['config']['labeled_category']
    labeled_dataset = CustomDataset(labeled_dataset_path, True, 
                                    labeled_category, data_transform)

    unlabeld_dataset_path = cfg['config']['unlabeled_data_path']
    unlabeld_dataset = CustomDataset(unlabeld_dataset_path, False,
                                    [], data_transform)
    
    random_seed = 42
    validation_ratio = 0.2
    dataset_size = len(labeled_dataset)
    dataset_indices = list(range(dataset_size))

    np.random.seed(random_seed)
    np.random.shuffle(dataset_indices)

    val_split_index = int(np.floor(dataset_size * validation_ratio))
    train_idx, val_idx = dataset_indices[val_split_index:], 
                        dataset_indices[:val_split_index]
    
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    batch_size = cfg['config']['batch_szie']
    dl = DataLoader(labeled_dataset, batch_size=batch_size,
                    sampler=train_sampler, num_workers=4)
    dtest = DataLoader(labeled_dataset, batch_size=batch_size,
                    sampler=val_sampler, num_workers=4)
    
    du = DataLoader(unlabeled_dataset, batch_size=batch_size,
                    num_workers=4)
    
    ceal_learning_algorithm(du=du, dl=dl, dtest=dtest)


if __name__ == '__main__':
    main()