import torch
from torch.utils.data.dataset import Dataset
from glob import glob
import math
import random
import numpy as np
import abc

class ClassicalMechanicsDataset(Dataset, metaclass=abc.ABCMeta):
    def __init__(self, path, data_file_indices=None):
        self.data_path = path

        path_glob = sorted(glob(self.data_path + '/*'))
        if type(data_file_indices) != type(None):
            self.data_files = np.take(path_glob, data_file_indices)
        else:
            self.data_files = path_glob
        
    
    def __len__(self):
        return len(self.data_files)
    
    
    @abc.abstractmethod
    def __getitem__(self, idx):
        pass

    
    @classmethod
    def train_test_split(dataset_cls, path, test_frac=0, random_state=None, max_actions=30000):        
        data_glob = glob(path + '/*')
        
        if not data_glob:
            dataset_cls(path).generate_data(max_actions=max_actions)
            data_glob = glob(path + '/*')
            
        
        test_size = int(math.floor(test_frac * len(data_glob)))

        print('Test size: ', test_size)

        if test_size != test_frac * len(data_glob):
            print(f'train_test_split response: test fraction rounded to ' +
                f'{test_size/len(data_glob)} ({test_size} simulations)')

        all_indices = np.array(range(len(data_glob)))
        
        # test_indices = random.choices(all_indices, k=test_size)
        # train_indices = [i for i in all_indices if i not in test_indices]
        
        # train_dataset = dataset_cls(path, data_file_indices=train_indices)
        # test_dataset = dataset_cls(path, data_file_indices=test_indices)

        test_indices = np.random.choice(all_indices, size=int(test_frac * len(all_indices)))

        mask = np.ones(len(all_indices), dtype=bool)
        mask[test_indices] = False

        train_indices=all_indices[mask]

        train_dataset = dataset_cls(path, data_file_indices=train_indices)
        test_dataset = dataset_cls(path, data_file_indices=test_indices)

        return (train_dataset, test_dataset)
    
    
    
    @abc.abstractmethod
    def generate_data(self, max_actions=30000): pass