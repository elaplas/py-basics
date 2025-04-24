import random
import math


class Perceptron:
    def __init__(self, n_inputs):
        self.W = [random.uniform(-1,1) for _ in range(n_inputs)]
        self.bias = random.uniform(-1,1)
    def __call__(self, X):
        if len(X) != len(self.W):
            raise "Input dimensions mismatch"
        res = 0.0
        for i in range(len(self.W)):
            res += self.W[i]*X[i]
        res += self.bias
        return math.tanh(res)

class Layer:
    def __init__(self, n_inputs, n_outputs):
        self.perceptrons = [Perceptron(n_inputs) for _ in range(n_outputs)]
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
    
    def __call__(self, X):
        if len(X) != self.n_inputs:
            raise "Input dimensions mismatch"
        
        res = []
        for i in range(self.n_outputs):
            res_i = self.perceptrons[i](X)
            res.append(res_i)
        return res
    
class MLP:
    def __init__(self, n_inputs, n_outputs):
        self.layers = []
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        while n_inputs//2 !=0 and n_inputs//2 !=1:
            self.layers.append(Layer(n_inputs, n_inputs//2))
            n_inputs = n_inputs//2
        self.layers.append(Layer(n_inputs, n_outputs))

    def __call__(self, X):
        if len(X) != self.n_inputs:
            raise "input dimension mismatch"
        cur_input = X
        cur_output = None
        for i in range(len(self.layers)):
            cur_output = self.layers[i](cur_input)
            cur_input = cur_output
        return cur_input

mlp = MLP(12, 3)
x = [1,2,3,4,5,6, 7,8,9,10,11,12]
o = mlp(x)
print(o)
