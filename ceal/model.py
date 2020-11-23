from typing import Optional, Callable

from torchvision.models import alexnet
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch

class AlexNet(object):
    def __init__(self, n_classes =2, device: Optional[str] = None):
        self.n_classes = n_classes 
        self.model = alexnet(pretrained=True, progress=True)

        self.__freeze_all_layers()
        self.__change_last_layers()
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def __freeze_all_layers(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False

    def __change_last_layers(self) -> None:
        self.model.classifier[6] = nn.Linear(4096, self.n_classes)

    def __add_softmax_layer(self) -> None:
        self.model = nn.Sequential(self.model, nn.LogSoftmax(dim=1))

    def __train_one_epoch(self, train_loader: DataLoader,
                            optimizer: optim,
                            criterion: Callable,
                            valid_loader: DataLoader = None,
                            epoch: int = 0,
                            each_batch_idx: int = 10) -> None:
        train_loss = 0
        data_size = 0
        
        for batch_idx, sample_batched in enumerate(train_loader):
            data, label = sample_batched['image'], sample_batched['label']
            
            data = data.to(self.device)
            data = data.float()
            label = label.to(self.device)

            optimizer.zero_grad()

            pred_prob = self.model(data)

            loss = criterion(pred_prob, label)
            loss.backward()

            train_loss += loss.item()
            data_size += label.size(0)

            optimizer.step()

            if batch_idx % each_batch_idx == 0:
                print('Train Epoch: {} [ {}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data),
                    len(train_loader.sampler.indices),
                    100. * (batch_idx * len(data)) / len(train_loader.sampler.indices),
                    loss.item()))
        
        if valid_loader:
            acc = self.evaluate(test_loader=valid_loader)
            print('Accuracy on the valid dataset {}'.format(acc))
        
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / data_size))
    
    def train(self, epochs: int, train_loader: DataLoader,valid_loader: DataLoader = None) -> None:
        self.model.to(self.device)
        self.model.train()
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr = 0.001, momentum = 0.9 )
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            self.__train_one_epoch(train_loader=train_loader,
                                    optimizer=optimizer,
                                    criterion = criterion,
                                    valid_loader = valid_loader,
                                    epoch=epoch)
    
    def evaluate(self, test_loader: DataLoader) -> float:
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, sample_batched in enumerate(test_loader):
                data, labels = sample_batched['image'], sample_batched['label']
                data = data.to(self.device)
                data = data.float()
                labels = labels.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total
    
    def predict(self, test_loader):
        
        self.model.eval()
        self.model.to(self.device)
        predict_results = np.empty(shape=(0,2))
        
        with torch.no_grad():
            for batch_idx, sample_batched in enumerate(test_loader):
                data, _ = sample_batched['image'], sample_batched['label']
                data = data.to(self.device)
                data = data.float()
                outputs = self.model(data)
                #outptus = softmax(outputs, dim=1)
                predict_results = np.concatenate((predict_results, outputs.cpu().numpy()))
        
        return predict_results
