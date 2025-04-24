import torch
import random


# __next__():
# - is called by next() and it perform only one iteration and return the result of it.
# - raises StopIteration exception when it is at the end of iteration
#
#__iter__():
# - is called by for loop and it performs the whole iterations if it is implemented independent from __next__() using yield
# - if __next__() is already implemented, __iter__() implementation is trivial and it is only the returning of "self"
#
class DataLoader:
    def __init__(self, x, y, batch_size, shuffle = True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.s = 0
        self.e = batch_size
        self.data_size = x.shape[0]
        self.indeces = [i for i in range( self.data_size )]
        if shuffle:
            random.shuffle(self.indeces)
    
    def __next__(self):  
        if self.s >= self.data_size:
            raise StopIteration
        batch_indeces = self.indeces[self.s:self.e]
        self.s += self.batch_size
        self.e += self.batch_size
        return self.x[batch_indeces], self.y[batch_indeces]
    
    def __iter__(self):
        return self
    
class DataLoader1:
    def __init__(self, x, y, batch_size, shuffle = True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.s = 0
        self.e = batch_size
        self.data_size = x.shape[0]
        self.indeces = [i for i in range( self.data_size )]
        if shuffle:
            random.shuffle(self.indeces)
    
    def __iter__(self):
        while self.s < self.data_size:
            batch_indeces = self.indeces[self.s:self.e]
            self.s += self.batch_size
            self.e += self.batch_size
            yield self.x[batch_indeces], self.y[batch_indeces]
    

X = torch.arange(90).reshape(10,3,3)
Y = torch.arange(10).reshape(10,1,1)

data_loader = DataLoader(X, Y, 2)

print("................DataLoader............")
for _ in range(2):
    x,y = next(data_loader)
    print("....next....")
    print(x)
    print(y)
    print("....next....")

for x, y in data_loader:
    print("....iter....")
    print(x)
    print(y)
    print("....iter....")
    print("........")

print("................DataLoader1............")
data_loader = DataLoader1(X, Y, 2)
for x, y in data_loader:
    print("....iter....")
    print(x)
    print(y)
    print("....iter....")
    print("........")