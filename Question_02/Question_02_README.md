Assumptions and design choices:
1. Network Architecture: We use 4 layers (3 hidden layers and 1 output layer) with sizes [784, 128, 64, 32, 10]. The input size is 784 (28x28 pixels for MNIST images), and the output size is 10 (for 10 digit classes).
2. Activation Functions: We use ReLU (Rectified Linear Unit) for hidden layers and Softmax for the output layer.
3. Loss Function: We use Cross-Entropy loss, which is suitable for multi-class classification problems.
4. Gradient Descent: We implement stochastic gradient descent (SGD) without mini-batches for simplicity. In practice, mini-batch SGD would be more efficient.
5. Initialization: We initialize weights with small random values and biases with zeros.
6. Learning Rate: We use a learning rate of 0.1, 0.001, 0.0001.( Tried all the 3 learning rate , 0.0001 gives low loss compared to others)
7. Epochs: Trained for 100,500 epochs. This number can be adjusted based on convergence and computational resources.
8. Data Preprocessing: We standardize the input features using StandardScaler from sklearn.

Instructions to Run the Project
1. Set Up the Environment
Ensure you have installed the required dependencies. You can set up the environment using the environment.yaml file:

```bash

conda env create -f environment.yaml

```
```bash

conda activate myenv  # Replace with your environment name if different

```
2. Running the Code
To execute the project, navigate to the directory containing mnist.py, and run the script:

```bash

python mnist.py

```
