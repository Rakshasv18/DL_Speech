import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from models.hiddenlayer import HiddenLayer
except ImportError:
    from hiddenlayer import HiddenLayer

class MultiLayerPerceptron:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.001):
        self.layers = []
        self.learning_rate = learning_rate
        
        # Input layer to first hidden layer
        self.layers.append(HiddenLayer(input_size, hidden_sizes[0], activation='relu'))
        
        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            self.layers.append(HiddenLayer(hidden_sizes[i-1], hidden_sizes[i], activation='relu'))
        
        # Last hidden layer to output layer
        self.layers.append(HiddenLayer(hidden_sizes[-1], output_size, activation='softmax'))
    
    def forward(self, X):
        a = X
        for layer in self.layers:
            a = layer.forward(a)
        return a
    
    def backward(self, y_pred, y_true):
        # Cross-Entropy Loss derivative
        da = y_pred - y_true
        
        for layer in reversed(self.layers):
            da, dW, db = layer.backward(da)
            layer.update(dW, db, self.learning_rate)
    
    def train(self, X, y, epochs=100):
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Compute Loss (Cross-Entropy Loss)
            loss = self.compute_loss(y_pred, y)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
            
            # Backward pass
            self.backward(y_pred, y)
    
    def compute_loss(self, y_pred, y_true):
        return -np.sum(y_true * np.log(y_pred + 1e-8)) / y_true.shape[0]
    
    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)
