import numpy as np

class Layer:
    def __init__(self, n_inputs, n_outputs):
        self.W = np.random.rand(n_outputs, n_inputs+1)*2-1
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
    
    def __call__(self, X):
        if len(X) != self.n_inputs:
            raise "Input dimensions mismatch"
        X = X + [1]
        X = np.array(X).reshape(self.n_inputs+1, 1)
        res = np.tanh(np.matmul(self.W, X))
        return [float(f) for f in np.squeeze(res)]
    
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
