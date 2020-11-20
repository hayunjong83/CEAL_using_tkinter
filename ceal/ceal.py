from model import AlexNet
from torch.utils.data import DataLoader
from torchvision import transforms
from model import AlexNet
from utils import get_uncertain_samples, get_high_confidence_samples, update_threshold
from utils import CatsAndDogs, Normalize, RandomCrop, SquarifyImage, ToTensor
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import shutil
import os


def ceal_learning_algorithm(du: DataLoader, 
                            dl: DataLoader,
                            dtest: DataLoader,
                            k : int = 20,
                            delta_0: float = 0.05,
                            dr : float = 0.0033,
                            t : int = 1,
                            epochs: int = 3,
                            criteria: str = 'cl',
                            max_iter: int = 45):
    model = AlexNet(n_classes =2, device=None)
    model.train(epochs = epochs, train_loader=dl, valid_loader=None)

    acc = model.evaluate(test_loader=dtest)
    print('====> Initial accuracy: {}'.format(acc))

    pred_prob = model.predict(test_loader=du)
    
    uncert_samp_idx, _ = get_uncertain_samples(pred_prob=pred_prob, k=k, criteria=criteria)
    print((uncert_samp_idx))
    # print(du.dataset[0])
    # for i, (images, labels) in enumerate(du):
    #     fname, _ = du.dataset[i]
    #     print(dir(fname))
    for idx in uncert_samp_idx:
        filename = du.dataset[idx]['filename']
        
        print(filename.split(os.sep)[-1])
        
        shutil.copy(filename, '../data/labeling_scheduled/'+ filename.split(os.sep)[-1])


    #uncert_samp_idx = [du.sampler.indices[idx] for idx in uncert_samp_idx]
    

    #hcs_idx, hcs_labels = get_high_confidence_samples(pred_prob=pred_prob,delta=delta_0)
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
    batch_size = 16
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


