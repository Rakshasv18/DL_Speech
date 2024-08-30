import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from models.multilayerperceptron import MultiLayerPerceptron

# Load and preprocess MNIST data
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = MinMaxScaler().fit_transform(X)  # Scale to [0, 1] range
y = y.astype(int)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert y_train and y_test to one-hot encoding
y_train_one_hot = np.zeros((y_train.size, 10))
y_train_one_hot[np.arange(y_train.size), y_train] = 1

y_test_one_hot = np.zeros((y_test.size, 10))
y_test_one_hot[np.arange(y_test.size), y_test] = 1

# Initialize and train the MLP
mlp = MultiLayerPerceptron(input_size=784, hidden_sizes=[128, 64, 32], output_size=10, learning_rate=0.0001)
mlp.train(X_train, y_train_one_hot, epochs=100)

# Evaluate on test data
y_test_one_hot_pred = mlp.forward(X_test)
predictions = np.argmax(y_test_one_hot_pred, axis=1)
accuracy = np.mean(predictions == y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
