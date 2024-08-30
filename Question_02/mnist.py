# import numpy as np
# from sklearn.datasets import fetch_openml
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# from models.multilayerperceptron import MultiLayerPerceptron

# # Load and preprocess MNIST data
# X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
# X = MinMaxScaler().fit_transform(X)  # Scale to [0, 1] range
# y = y.astype(int)

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Convert y_train and y_test to one-hot encoding
# y_train_one_hot = np.zeros((y_train.size, 10))
# y_train_one_hot[np.arange(y_train.size), y_train] = 1

# y_test_one_hot = np.zeros((y_test.size, 10))
# y_test_one_hot[np.arange(y_test.size), y_test] = 1

# # Initialize and train the MLP
# mlp = MultiLayerPerceptron(input_size=784, hidden_sizes=[128, 64, 32], output_size=10, learning_rate=0.0001)
# mlp.train(X_train, y_train_one_hot, epochs=100)

# # Evaluate on test data
# y_test_one_hot_pred = mlp.forward(X_test)
# predictions = np.argmax(y_test_one_hot_pred, axis=1)
# accuracy = np.mean(predictions == y_test)
# print(f"Test Accuracy: {accuracy * 100:.2f}%")

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from models.multilayerperceptron import MultiLayerPerceptron

def load_and_preprocess_data():
    """
    Load and preprocess the MNIST dataset.

    - Fetches the MNIST dataset from OpenML.
    - Scales the features to the [0, 1] range using MinMaxScaler.
    - Converts target labels to integers.

    Returns:
        tuple: (X, y) where X is the feature matrix and y is the target vector.
    """
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    X = MinMaxScaler().fit_transform(X)  # Scale to [0, 1] range
    y = y.astype(int)
    return X, y

def split_data(X, y):
    """
    Split the dataset into training and test sets.

    Args:
        X (np.ndarray): The feature matrix.
        y (np.ndarray): The target vector.

    Returns:
        tuple: (X_train, X_test, y_train, y_test) where
            X_train and X_test are feature matrices for training and test sets,
            y_train and y_test are target vectors for training and test sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def convert_to_one_hot(y):
    """
    Convert target vector to one-hot encoding.

    Args:
        y (np.ndarray): The target vector.

    Returns:
        np.ndarray: The one-hot encoded target matrix.
    """
    one_hot = np.zeros((y.size, 10))
    one_hot[np.arange(y.size), y] = 1
    return one_hot

def main():
    """
    Main function to execute the workflow:
    - Load and preprocess the MNIST dataset.
    - Split the dataset into training and test sets.
    - Convert target vectors to one-hot encoding.
    - Initialize and train a MultiLayerPerceptron (MLP) model.
    - Evaluate the model on the test data and print the accuracy.
    """
    # Load and preprocess MNIST data
    X, y = load_and_preprocess_data()

    # Split the data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Convert y_train and y_test to one-hot encoding
    y_train_one_hot = convert_to_one_hot(y_train)
    y_test_one_hot = convert_to_one_hot(y_test)

    # Initialize and train the MLP
    mlp = MultiLayerPerceptron(input_size=784, hidden_sizes=[128, 64, 32], output_size=10, learning_rate=0.0001)
    mlp.train(X_train, y_train_one_hot, epochs=100)

    # Evaluate on test data
    y_test_one_hot_pred = mlp.forward(X_test)
    predictions = np.argmax(y_test_one_hot_pred, axis=1)
    accuracy = np.mean(predictions == y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()

