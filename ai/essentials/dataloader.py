import numpy as np
import torch
import random


class DataSet:
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
    
    def __getitem__(self, idx):    
        if type(self.x_data) == np.ndarray and type(self.y_data) == np.ndarray:
            ## Repair the shape of the indexed array after indexing (after indexing the first elment of shape is removed)
            return self.x_data[idx].reshape((1,)+self.x_data[idx].shape), self.y_data[idx].reshape((1,)+self.y_data[idx].shape)
        elif type(self.x_data) == torch.Tensor and type(self.y_data) == torch.Tensor:
            ## Repair the shape of the indexed tensor after indexing (after indexing the first elment of shape is removed)
            return self.x_data[idx].reshape((1,)+self.x_data[idx].shape), self.y_data[idx].reshape((1,)+self.y_data[idx].shape)
        elif type(self.x_data) == list and type(self.y_data) == list:
            x = self.x_data[idx]
            y = self.y_data[idx]
            if type(x) != list:
                x = [x]
            if type(y) != list:
                y = [y]
            return x, y
        else:
            raise Exception("data type us not supported")

        
    def __len__(self):
        if type(self.x_data == list) and type(self.y_data == list):
            assert len(self.x_data) == len(self.y_data), "data and its labels have different sizes!"
            return len(self.x_data)
        elif type(self.x_data) == np.ndarray and type(self.y_data)==np.ndarray:
            assert self.x_data.size() == self.y_data, "data and its labels have different dimensions!"
            return self.x_data.shape[0]
        elif type(self.x_data) == torch.tensor and type(self.y_data) == torch.tensor:
            assert self.x_data.size() == self.y_data, "data and its labels have different dimensions!"
            return self.x_data.shape[0]
        else:
            raise Exception("dataset type is not supported")
        

class DataLoader:
    def __init__(self, dataset: DataSet, batch_size, shuffle):
        self.dataset = dataset
        self.batch_size = batch_size
        self.sampler = [i for i in range(len(self.dataset))]
        if shuffle:
            random.shuffle(self.sampler)
    
    def __get_batch_indecies(self):
        batch_indecies = []
        for idx in self.sampler:
            batch_indecies.append(idx)
            if len(batch_indecies) == self.batch_size:
                yield batch_indecies
                batch_indecies = []
        return batch_indecies

    def __iter__(self):
        for batch_indecies in self.__get_batch_indecies():
            X = torch.tensor([])
            Y = torch.tensor([])
            for idx in batch_indecies:
                x, y = self.dataset[idx]
                if type(x) == np.ndarray or type(x) == list:
                    x = torch.tensor(x)
                if type(y) == np.ndarray or type(y) == list:
                    y = torch.tensor(y)
                ## Add the batch dimension
                x = x.reshape((1,) + x.shape)
                y = y.reshape((1,) + y.shape)
                assert type(x) == torch.Tensor, "dataset type is not supported"
                assert type(y) == torch.Tensor, "dataset type is not supported"
                ## Concatenate tensors to form the batch
                X = torch.cat((X, x), 0)
                Y = torch.cat((Y, y), 0)
            yield X, Y
        return X, Y
    

X = np.random.randint(-5, 5, (20, 2))
Y = np.random.randint(0, 2, (20, 1))
dataset = DataSet(X, Y)
data_loader = DataLoader(dataset, 5, True)
for x, y in data_loader:
    print(x.shape)
    print(y.shape)


