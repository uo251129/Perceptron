import os
os.chdir("\\".join(os.path.realpath(__file__).split("\\")[:-1])) 
from perceptron import Perceptron
import numpy as np
import math

""" Test function that generates a random dataset and a random linear function that determines the class of each 
element. This functions takes the following parameters:
dataset_size: number of elements that will be generated for the dataset.
number_of_variables: dimensionality of the dataset.
test_split: Percentage within 0 and 1 that determines the size of the training and testing sets.
number_of_iterations: Number of times that the training process will be repeated for the preceptron.
target_type: If "classification", the generated target variable will be discrete. If "regression", it will be 
            continous.
learning_rate: Meta-parameter of the perceptron model.
verbose: If set to false, the function will not print anything in the console. 
"""
def test_perceptron(dataset_size, number_of_variables, test_split, number_of_iterations, 
                                      target_type, learning_rate = 0.05, verbose = True):
    m = np.random.uniform(-1, 1, number_of_variables)
    b = np.random.uniform(-1, 1)
    
    linear_function = lambda x : np.dot(m, x) + b
    
    inputs = [np.random.uniform(-1, 1, number_of_variables) for i in range(0, dataset_size)]
    
    if target_type == "classification":
        targets = [int(linear_function(input) > 0) for input in inputs]
    elif target_type == "regression":
        targets = [linear_function(input) for input in inputs]
        
    
    x_train = inputs[0:math.floor(test_split * dataset_size)]
    x_test = inputs[math.floor(test_split * dataset_size):dataset_size]
    
    y_train = targets[0:math.floor(test_split  * dataset_size)]
    y_test = targets[math.floor(test_split * dataset_size):dataset_size]
    
    perceptron = Perceptron(number_of_variables, learning_rate, target_type)
    
    train_accuracy = perceptron.train(x_train, y_train, number_of_iterations)
    
    if verbose: print("Train accuracy at each iteration: %s" % train_accuracy)
    
    test_output = perceptron.test(x_test)
    
    test_accuracy = perceptron.val_function(y_test, test_output)
    
    if verbose: print("Test accuracy: %s" % test_accuracy)
    
    return train_accuracy, test_accuracy
