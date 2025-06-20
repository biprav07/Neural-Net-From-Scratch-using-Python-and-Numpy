import numpy as np

class ReLU:
    def forward(self, input_data):
        self.input = input_data
        return np.maximum(0, input_data)

    def backward(self, output_gradient, learning_rate):
        return output_gradient * (self.input > 0)

class Sigmoid:
    def forward(self, input_data):
        self.output = 1 / (1 + np.exp(-input_data))
        return self.output

    def backward(self, output_gradient, learning_rate):
        return output_gradient * (self.output * (1 - self.output))
