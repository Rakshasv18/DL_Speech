import numpy as np

class HiddenLayer:
    def __init__(self, input_size, output_size, activation='relu'):
        self.input_size = input_size
        self.output_size = output_size
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)  # He Initialization
        self.b = np.zeros((1, output_size))
        self.activation = activation
        self.input = None
        self.z = None
        self.a = None
        
    def forward(self, X):
        self.input = X
        self.z = np.dot(X, self.W) + self.b
        if self.activation == 'relu':
            self.a = np.maximum(0, self.z)
        elif self.activation == 'softmax':
            exp_z = np.exp(self.z - np.max(self.z, axis=1, keepdims=True))
            self.a = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return self.a
    
    def backward(self, da):
        if self.activation == 'relu':
            dz = da * (self.z > 0)  # Derivative of ReLU
        elif self.activation == 'softmax':
            dz = da  # No activation derivative needed here; already handled in softmax
        
        dW = np.dot(self.input.T, dz)
        db = np.sum(dz, axis=0, keepdims=True)
        da_prev = np.dot(dz, self.W.T)
        return da_prev, dW, db
    
    def update(self, dW, db, learning_rate):
        self.W -= learning_rate * dW
        self.b -= learning_rate * db
