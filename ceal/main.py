import os
import numpy as np

from ceal import ceal
from custom_dataset import make_labeled_dataloader
from custom_dataset import make_unlabeled_dataloader

def main():

    configuration_path = 'configuration.yml' 
    if not os.path.isfile(configuration_path):
        print('There is no configuration file. Please Check it!')
        return
    
    dl, dtest = make_labeled_dataloader(configuration_path)
    du = make_unlabeled_dataloader(configuration_path)
    ceal(du=du, dl=dl, dtest=dtest, configuration_path=configuration_path)

if __name__ == '__main__':
    main()