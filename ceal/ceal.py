import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, models
import numpy as np
import shutil
import os
import copy
import yaml

from utils import get_uncertain_samples, get_high_confidence_samples, update_threshold
from labeling import labeling
from custom_dataset import make_labeled_dataloader
from custom_dataset import make_unlabeled_dataloader


def train(model, device, train_loader, val_loader, epochs, criterion, optimizer):
    
    best_loss = 100000.0
    best_model_wts = copy.deepcopy(model.state_dict())
    model.train()
    for epoch in range(epochs):
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
    
    model.load_state_dict(best_model_wts)
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
                out = F.softmax(out, dim=0).tolist()
                prob_list.append(out)
    return prob_list        

def ceal(du, dl, dtest, configuration_path):
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
    elif model == 'DenseNet':
        model_ft = models.densenet161(pretrained=True)
        for param in model_ft.parameters():
            param.requires_grad = False
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, len(labeled_category))
    else:
        print("Model is not defined yet.")
        return
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model_ft.parameters(), lr = 0.001, betas=(0.9, 0.999))
    num_epochs = cfg['config']['epochs']

    max_iteration = cfg['config']['max_iteration']

    uncertain_samples =[]
    high_confidence_samples = []

    for iteration in range(max_iteration):
        if len(high_confidence_samples) != 0:
            dl, dtest = make_labeled_dataloader(configuration_path)

        model_ft = train(model_ft, device, dl, dtest, num_epochs, criterion, optimizer_ft)

        if len(high_confidence_samples) != 0:
            for sample in high_confidence_samples:
                filename = sample[0]
                label = labeled_category[sample[1]]
                shutil.move(os.path.join(labeled_dataset_path, label, filename.split(os.sep)[-1]), filename)
            
            du = make_unlabeled_dataloader(configuration_path)
            high_confidence_samples.clear()

        pred_prob = predict(model_ft, device, du)

        k = cfg['ceal']['k']
        criteria = cfg['ceal']['criteria']
    
        uncert_samp_idx, _ = get_uncertain_samples(pred_prob=pred_prob, k=k, criteria=criteria)
        scheduling_path = cfg['config']['labeling_scheduling_path']

        if not os.path.isdir(scheduling_path):
            os.makedirs(scheduling_path, exist_ok=True)
        for idx in uncert_samp_idx:
            filename = du.dataset[idx]['filename']    
            uncertain_samples.append(filename)
            shutil.copy(filename, scheduling_path+ '/'+ filename.split(os.sep)[-1])

        labeling(scheduling_path, labeled_category)
        
        delta_0 = cfg['ceal']['delta_0']
        hcs_idx, hcs_labels = get_high_confidence_samples(pred_prob=pred_prob,delta=delta_0)
        
        for idx, label in zip(hcs_idx, hcs_labels):
            filename = du.dataset[idx]['filename']
            high_confidence_samples.append([filename, label])
        
        labeled_dataset_path = cfg['config']['labeled_data_path']

        for sample in high_confidence_samples:
            filename = sample[0]
            label = labeled_category[sample[1]]
            shutil.move(filename, os.path.join(labeled_dataset_path, label, filename.split(os.sep)[-1]))
        for sample in uncertain_samples:
            os.remove(sample)
        uncertain_samples.clear()
    
    for sample in high_confidence_samples:
        filename = sample[0]
        label = labeled_category[sample[1]]
        shutil.move(os.path.join(labeled_dataset_path, label, filename.split(os.sep)[-1]), filename)