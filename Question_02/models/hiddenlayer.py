import numpy as np

class HiddenLayer:
    def __init__(self, input_size, output_size, activation='relu'):
        """
        Initialize a hidden layer of the neural network.

        Args:
            input_size (int): Number of input features.
            output_size (int): Number of neurons in the layer.
            activation (str): Activation function to use ('relu' or 'softmax').
        """
        self.input_size = input_size
        self.output_size = output_size
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)  # He Initialization
        self.b = np.zeros((1, output_size))
        self.activation = activation
        self.input = None
        self.z = None
        self.a = None
        
    def forward(self, X):
        """
        Perform the forward pass of the hidden layer.

        Args:
            X (np.ndarray): Input data matrix of shape (n_samples, input_size).

        Returns:
            np.ndarray: Output of the layer after activation.
        """
        self.input = X
        self.z = np.dot(X, self.W) + self.b
        if self.activation == 'relu':
            self.a = np.maximum(0, self.z)
        elif self.activation == 'softmax':
            exp_z = np.exp(self.z - np.max(self.z, axis=1, keepdims=True))
            self.a = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return self.a
    
    def backward(self, da):
        """
        Perform the backward pass of the hidden layer.

        Args:
            da (np.ndarray): Gradient of the loss with respect to the layer's output.

        Returns:
            tuple: (da_prev, dW, db) where
                da_prev (np.ndarray): Gradient of the loss with respect to the previous layer's output.
                dW (np.ndarray): Gradient of the loss with respect to the layer's weights.
                db (np.ndarray): Gradient of the loss with respect to the layer's biases.
        """
        if self.activation == 'relu':
            dz = da * (self.z > 0)  # Derivative of ReLU
        elif self.activation == 'softmax':
            dz = da  # No activation derivative needed here; already handled in softmax
        
        dW = np.dot(self.input.T, dz)
        db = np.sum(dz, axis=0, keepdims=True)
        da_prev = np.dot(dz, self.W.T)
        return da_prev, dW, db
    
    def update(self, dW, db, learning_rate):
        """
        Update the layer's weights and biases using gradient descent.

        Args:
            dW (np.ndarray): Gradient of the loss with respect to the weights.
            db (np.ndarray): Gradient of the loss with respect to the biases.
            learning_rate (float): Learning rate for the update.
        """
        self.W -= learning_rate * dW
        self.b -= learning_rate * db
