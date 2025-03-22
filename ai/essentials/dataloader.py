import os
from PIL import Image
import numpy as np
import torch
import random

class Dataset:
    def __init__(self, data):
        self.data = data
        self.element_shape = None
    def __getitem__(self, idx):
        tmp = self.data[idx]
        if type(self.data[idx]) !=list and type(self.data[idx])!=np.array or type(self.data[idx])!=torch.tensor:
            tmp = [tmp]
        return tmp
    def __len__(self):
        if type(self.data) == list:
            return len(self.data)
        else:
            return self.data.shape[0]

class DataLoader:
    def __init__(self, dataset: Dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.sampler = None
        if shuffle:
            self.sampler = [i for i in range(len(self.dataset))]
            random.shuffle(self.sampler)
        else:
            self.sampler = range(0, len(self.dataset))
    def __get_batch(self):
        batch = []
        for i in self.sampler:
            batch.append(i)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
    def __iter__(self):
        batch_data = []
        for batch in self.__get_batch():
            for i in batch:
                tmp = self.dataset[i]
                batch_data = torch.cat((torch.tensor(batch_data), torch.tensor(tmp)), 0)
            yield batch_data
            batch_data = []

L = [1,2,3,4,5,6,7,8,9,10,11,12]
dataset = Dataset(L)
dataloader = DataLoader(dataset=dataset, batch_size=3)
for batch in dataloader:
    print(batch)