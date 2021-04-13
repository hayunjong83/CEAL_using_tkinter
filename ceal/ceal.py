from model import AlexNet
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import transforms, models
from model import AlexNet
from utils import get_uncertain_samples, get_high_confidence_samples, update_threshold
from utils import CatsAndDogs, Normalize, RandomCrop, SquarifyImage, ToTensor
import labeling
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import shutil
import os
import copy

def train(model, device, train_loader, val_loader, epochs, criterion, optimizer):
    
    best_loss = 100000.0
    best_model_wts = copy.deepcopy(model.state_dict())
    model.train()
    for epoch in epochs:
        print('Epoch [{}/{}]'.format(epoch+1, epochs))
        
        train_loss = 0
        train_acc = 0.0
        train_size = 0

        for i, sample in enumerate(train_loader):
            data, labels = sample['image'], sample['label']
            
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)
            train_acc += torch.sum(preds == labels.data)
            train_size += labels.size(0)
        
        train_loss /= train_size
        train_acc /= train_size
        print('    train Loss {:.4f}, Accuracy:{:.4f}'.format(train_loss, train_acc))

        val_loss = 0
        val_acc = 0.0
        val_size = 0

        with torch.no_grad():
            model.eval()
            for i, sample in enumerate(val_loader):
                data, labels = sample['image'], sample['label']
            
                data = data.to(device)
                labels = labels.to(device)

                outputs = model(data)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * data.size(0)
                val_acc += torch.sum(preds == labels.data)
                val_size += labels.size(0)
            
            val_loss /= val_size
            val_acc /= val_size
            print('    validation Loss {:.4f}, Accuracy:{:.4f}'.format(val_loss, val_acc))

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
    
    model.load_state_dice(best_model_wts)
    return model
def predict(model, device, uncertain_dataloader):
    model.eval()
    prob_list = []

    with torch.no_grad():
        for i, sample in enumerate(uncertain_dataloader):
            data = sample['image']
            data = data.to(device)

            outputs = model(data)
            for out in outputs:
                F.softmax(out, dim=0).tolist()
                prob_list.append(out)

    return prob_list        

def ceal(du, dl, dtest):
    configuration_path = 'configuration.yml'
    if not os.path.isfile(configuration_path):
        print('There is no configuration file. Please Check it!')
        return
    
    with open(configuration_path, 'r') as cfg_path:
        cfg = yaml.safe_load(cfg_path)
    
    model = cfg['config']['model']
    labeled_category = cfg['config']['labeled_category']

    if model == 'AlexNet':
        model_ft = models.AlexNet(pretrained=True)
        for param in model_ft.parameters():
            param.requires_grad = False
        model_ft.classifier[6] = nn.Linear(4096, len(labeled_category))
    elif model == 'ResNet':
        model_ft = models.resnet18(pretrained=True)
        for param in model_ft.parameters():
            param.requires_grad = False
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, len(labeled_category))
    elif model == 'DensNet':
        model_ft = models.densenet161(pretrained=True)
        for param in model_ft.parameters():
            param.requires_grad = False
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, len(labeled_category))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model_ft.parameters(), lr = 0.001, betas=(0.9, 0.999))
    num_epochs = cfg['config']['epochs']

    max_iteration = cfg['config']['max_iteration']

    for iteration in range(max_iteration):
        model_ft = train(model_ft, device, dl, dtest, num_epochs, criterion, optimizer_ft)
        pred_prob = predict(model_ft, device, du)

def ceal_learning_algorithm(du: DataLoader, 
                            dl: DataLoader,
                            dtest: DataLoader,
                            k : int = 20,
                            delta_0: float = 0.0005,
                            dr : float = 0.0033,
                            epochs: int = 10,
                            criteria: str = 'cl',
                            max_iter: int = 1):
    model = AlexNet(n_classes =2, device=None)
    val_acc_list = []

    for iteration in range(max_iter):
        model.train(epochs = epochs, train_loader=dl, valid_loader=dtest)
        acc = model.evaluate(test_loader=du)
        if iteration == 0:
            print('====> Initial accuracy: {}'.format(acc))
        else :
            print('=> #', iteration + 1, ' accuracy: ', acc)
        val_acc_list.append(acc)
        
        pred_prob = model.predict(test_loader=du)

        uncert_samp_idx, _ = get_uncertain_samples(pred_prob=pred_prob, k=k, criteria=criteria)
        #print((uncert_samp_idx))
        for idx in uncert_samp_idx:
            filename = du.dataset[idx]['filename']    
            #print(filename.split(os.sep)[-1])
            shutil.copy(filename, '../data/labeling_scheduled/'+ filename.split(os.sep)[-1])

        #labeling.main()

        #hcs_idx, hcs_labels = get_high_confidence_samples(pred_prob=pred_prob,delta=delta_0)

        #print(len(hcs_idx))        
        # for idx, label in zip(hcs_idx, hcs_labels):
        #     filename = du.dataset[idx]['filename']
        #     print(filename, label)
            

    #hcs_idx = [du.sampler.indices[idx] for idx in hcs_idx]
    #print(hcs_idx)

if __name__ == "__main__":

    dataset_train = CatsAndDogs(
        root_dir="../data/dl",
        labeled=True,
        transform=transforms.Compose(
            [SquarifyImage(),
             RandomCrop(224),
             Normalize(),
             ToTensor()]))

    dataset_test = CatsAndDogs(
        root_dir="../data/du",
        labeled=False,
        transform=transforms.Compose(
            [SquarifyImage(),
             RandomCrop(224),
             Normalize(),
             ToTensor()]))

    # Creating data indices for training and validation splits:
    random_seed = 123
    validation_split = 0.1  # 10%
    shuffling_dataset = True
    batch_size = 4
    dataset_size = len(dataset_train)
    
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    
    if shuffling_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]


    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    dl = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                     sampler=train_sampler, num_workers=4)
    dtest = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                     sampler=valid_sampler, num_workers=4)
    du = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                        num_workers=4)
    ceal_learning_algorithm(du=du, dl=dl, dtest=dtest)