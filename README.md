# Flawless_assignment

Project Overview

This repository contains two separate deep learning projects, each organized into its own folder. Both projects involve training and evaluating neural networks, and each folder includes the necessary files to run the respective models.

Question_01\
This folder contains all files related to the first deep learning model.

model_1: This directory contains the model's scripts and architecture definitions.\
  1. config.yaml: Configuration file for the model training process. Includes hyperparameters like learning rate, batch size, and the number of epochs.\
  2. dataloader.py: Script to load and preprocess the dataset.\
  3. model.py: Defines the neural network architecture and forward/backward propagation methods.\
  4. runs: Directory that stores the training logs, model checkpoints, and visualizations (e.g., TensorBoard files).\
  5. train.py: Main training script that initializes the model, loads the data, and trains the model.\

Question_01.readme: Detailed documentation of the first project, including model architecture, data used, and training methodology.

Question_02\
This folder contains files related to the second deep learning model, focused on solving the MNIST classification problem using a custom multi-layer perceptron.

mnist.py: Script that loads the MNIST dataset and manages preprocessing tasks.\
models: This directory contains all model-related Python files.\
__init__.py: Initializes the models module.\
hiddenlayer.py: Defines the custom HiddenLayer class used in the neural network.\
multilayerperceptron.py: Defines the MultiLayerPerceptron class, which builds and manages the overall network.\
Question_02.readme: Detailed documentation of the second project, including model architecture, data used, and training methodology.



Environment Setup\
To set up the environment required for both projects, use the environment.yaml file included in the repository. This file contains all the dependencies and versions required to run the code.

To create the environment, run the following command:

```bash
conda env create -f environment.yaml

```

Once the environment is created, activate it using:

```bash

conda activate <environment_name>

```

Replace <environment_name> with the name of the environment specified in environment.yaml.


