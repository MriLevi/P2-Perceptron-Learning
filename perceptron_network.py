import math

class Perceptron:

    def __init__(self, threshold, activation_function, weights):
        self.bias = -threshold
        self.threshold = threshold
        self.activation_function = activation_function
        self.weights = weights
        self.error_total = 0
        self.updatecount = 0
        self.MSE = None

    def __str__(self):
        return f'bias: {self.bias} - threshold: {self.threshold} - activation_function: {self.activation_function.__name__}\n' \
               f'weights: {self.weights}'

    def dot(self, x1, x2):
        """returns the inner product of two vectors"""
        """NOTE: Some checking should be done to verify the dimensions of the input - this is lazy"""
        return sum([x * y for x, y in zip(x1, x2)])

    def determine_output(self, inputs):
        """Determines the output of a perceptron by adding the dot product of the weights and inputs and adding bias"""
        return self.activation_function(self.bias + self.dot(self.weights, inputs))

    def update(self, inputs, target, learning_rate=0.1):
        """This function allows os to train/fit our perceptron by following the perceptron learning rule."""

        y = self.determine_output(inputs)
        error = target - y
        if error != 0:
            delta_weights = []
            for w,i in zip(self.weights, inputs):
                delta_weights.append(w + (learning_rate * error * i))
            delta_bias = learning_rate * error
            self.weights = delta_weights
            self.bias += delta_bias
            self.threshold = -self.bias
        #here we keep track of the updates we've done and the squared error, so we can easily determine the loss later
        self.updatecount += 1
        self.error_total += error**2

    def loss(self):
        """This function keeps track of the MSE by simply dividing the sum of squared errors,
        divided by the number of updates already done."""
        self.MSE = self.error_total/self.updatecount
        return self.MSE

class PerceptronLayer:

    def __init__(self, perceptrons):
        self.perceptrons = perceptrons

    def determine_outputs(self, inputs):
        """Determines the output for each perceptron by calling determine_output() for each perceptron
           Returns a list of output values."""
        outputs = [perceptron.determine_output(inputs) for perceptron in self.perceptrons]
        return outputs


class PerceptronNetwork:

    def __init__(self, perceptron_layers):
        self.perceptron_layers = perceptron_layers

    def feed_forward(self, initial_input):
        """Feed forward determines the output for each layer by using the output of the previous as input for the next
            It determines the first layers outputs outside of the loop to be able to then loop over the remaining
            layers. Finally, it returns the output of the final layer.
            NOTE: The first layer must always have its inputs defined."""
        inputs = self.perceptron_layers[0].determine_outputs(initial_input)
        for perceptron_layer in self.perceptron_layers[1:]:
            inputs = perceptron_layer.determine_outputs(inputs)
        result = inputs
        return result
