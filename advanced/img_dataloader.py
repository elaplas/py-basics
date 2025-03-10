import os
from PIL import Image
import numpy as np
import torch
import random

class Dataset:
    def __init__(self, files_dir="", resize=(28,28)):
        self.img_dir = files_dir
        self.resize = resize
        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        self.img_names = [ os.path.join(self.img_dir, f) for f in os.listdir(self.img_dir) if f.lower().endswith(extensions)]
    def __getitem__(self, idx):
        img = Image.open(self.img_names[idx])
        img = img.resize(self.resize)
        return np.array(img, dtype=np.uint8)
    def __len__(self):
        return len(self.img_names)

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