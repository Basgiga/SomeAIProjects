import numpy as np


# Define sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class Perceptron2:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        # Initialize weights and biases
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Weights and biases for layers
        self.W1 = np.random.rand(self.input_size, self.hidden_size)
        self.b1 = np.random.rand(1, self.hidden_size)
        self.W2 = np.random.rand(self.hidden_size, self.output_size)
        self.b2 = np.random.rand(1, self.output_size)

    def forward(self, X):
        # Forward propagation
        self.hidden_input = np.dot(X, self.W1) + self.b1
        self.hidden_output = sigmoid(self.hidden_input)

        self.output_input = np.dot(self.hidden_output, self.W2) + self.b2
        self.predicted_output = sigmoid(self.output_input)

        return self.predicted_output

    def backward(self, X, y):
        # Compute the loss (mean squared error)
        loss = np.mean((y - self.predicted_output) ** 2)

        # Backpropagation
        output_error = y - self.predicted_output
        output_delta = output_error * sigmoid_derivative(self.predicted_output)

        hidden_error = output_delta.dot(self.W2.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)

        # Update weights and biases using gradient descent
        self.W2 += self.hidden_output.T.dot(output_delta) * self.learning_rate
        self.b2 += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        self.W1 += X.T.dot(hidden_delta) * self.learning_rate
        self.b1 += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate

        return loss

    def train(self, X, y, epochs=10000):
        # Train the network
        for epoch in range(epochs):
            self.forward(X)
            loss = self.backward(X, y)

            # Print the loss every 1000 epochs to monitor progress
            if epoch % 1000 == 0:
                print(f"Epoch {epoch} - Loss: {loss}")

    def predict(self, X):
        # Predict the output for new data
        return self.forward(X)


# XOR data (inputs and expected outputs)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0], [1], [1], [0]])  # XOR output

# Initialize the Perceptron2 (with 2 inputs, 2 hidden neurons, and 1 output)
perceptron = Perceptron2(input_size=2, hidden_size=2, output_size=1, learning_rate=0.1)

# Train the network
perceptron.train(X, y, epochs=10000)

# Test the network
print("\nFinal output after training:")
print(perceptron.predict(X))