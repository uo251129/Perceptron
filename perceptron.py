import numpy as np

class Perceptron:
    eps = 1e-4
    threshold = 0
    
    """ Initializes a perceptron with the following meta-parameters:
    number_of_variables: Number of inputs of each element in the dataset that is going to be used to train the 
                         perceptron.
    learning_rate: Variable used to update the weights of the perceptron when an element is missclassified.
    target_type: "classification" for discrete targets and "regression" for continuous targets.
    """
    def __init__(self, number_of_variables, learning_rate, target_type):
        self.initialize_weights(number_of_variables)
        self.learning_rate = learning_rate
        self.target_type = target_type
        if target_type == "classification":
            self.activation_function = self.classification_activation
            self.val_function = self.accuracy
        elif target_type == "regression":
            self.activation_function = self.reggression_activation
            self.val_function = self.rmse
        else:
            raise ValueError("target_type should be 'classification' of 'reggression'")
        
    """ Initializes a set of weights with random values in the range of [-eps, eps]"""
    def initialize_weights(self, number_of_variables):
        self.weights = np.random.uniform(-self.eps, self.eps, number_of_variables + 1)
    
    """ Activation function used in regression problems that returns a float number."""
    def reggression_activation(self, input):
        return np.dot(np.append(1, input), self.weights)
        
    """ Activation function that classifies an element returning 0 or 1."""
    def classification_activation(self, input):
        return int(np.dot(np.append(1, input), self.weights) > self.threshold)    
    
    """ Update function that adjusts the set of weights to adjust the perceptron to a dataset."""
    def update_weights(self, input, target, output):
        input_with_bias = np.append(1, input) 
        self.weights = [self.weights[i] + self.learning_rate * (target - output) * input_with_bias[i]
                        for i in range(0, len(self.weights))]
    
    """ Error function used in classification models."""
    def accuracy(self, target, output):
        return sum([int(target[i]==output[i]) for i in range(len(target))]) / len(target)
    
    """ Error function used in reggression models."""
    def rmse(self, target, output):
        return np.sqrt(sum([(target[i] - output[i]) ** 2 for i in range(len(target))]) / len(target))        
    
    """" Train function that fits the perceptron to the dataset. A number of iterations can be indicated in order to
    repeat the training process multiple times. This function returns a list with the accuracy obtained in each
    iteration, in case that the target is discrete, and a list with the RMSE value in each iteration in case that the
    target is continuous.
    """
    def train(self, inputs, targets, number_of_iterations = 1):
        learning_curve = []
        for iteration in range(number_of_iterations):
            outputs = []
            for i in range(0, len(inputs)):
                output = self.activation_function(inputs[i])
                self.update_weights(inputs[i], targets[i], output)
                outputs = outputs + [output]
            learning_curve = learning_curve + [self.val_function(targets, outputs)]
        return learning_curve
    
    """ Returns the model's activation function for each element of a given dataset."""
    def test(self, inputs):
        return [self.activation_function(input) for input in inputs]