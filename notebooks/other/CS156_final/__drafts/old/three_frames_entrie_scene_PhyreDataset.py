import torch
from torch.utils.data.dataset import Dataset
from glob import glob
import math
import random
import numpy as np

class PhyreSequentialDataset(Dataset):
    @staticmethod
    def train_test_split(path, test_frac=0):
        data_glob = glob(path + '/*')
        
        test_size = int(math.floor(test_frac * len(data_glob)))

        if test_size != test_frac * len(data_glob):
            print(f'train_test_split response: test fraction rounded to ' +
                f'{test_size/len(data_glob)} ({test_size} simulations)')

        all_indices = list(range(len(data_glob)))
        test_indices = random.choices(all_indices, k=test_size)
        train_indices = [i for i in all_indices if i not in test_indices]

        # returns tupple(train_dataset, test_dataset)
        return (PhyreSequentialDataset(path, data_file_indices=train_indices), \
                PhyreSequentialDataset(path, data_file_indices=test_indices))



    def __init__(self, path, data_file_indices=None):
        self.data_path = path + '/*'
        self.data_file_indices = data_file_indices

        self.inputs = []
        self.outputs = []

        self.process_data()


    def __len__(self):
        return len(self.inputs)


    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


    def process_data(self):
        path_glob = sorted(glob(self.data_path))
        
        for file_idx in range(len(path_glob)):
            if file_idx not in self.data_file_indices: continue

            data = np.load(path_glob[file_idx])
            
            input_instance = []
            output_instance = []
            for frame_number in range(len(data)-3):
                temp_input = []
                temp_input.append(data[frame_number])
                temp_input.append(data[frame_number+1])
                temp_input.append(data[frame_number+2])

                input_instance.append(np.array(temp_input).flatten())
                output_instance.append(np.array(data[frame_number+3])[2,0:2].flatten())

            input_instance = torch.FloatTensor(np.array(input_instance))
            output_instance = torch.FloatTensor(np.array(output_instance))


            self.inputs.append(input_instance)
            self.outputs.append(output_instance)
            