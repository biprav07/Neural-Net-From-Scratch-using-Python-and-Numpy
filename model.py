class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self, X, y, epochs, learning_rate, loss_fn, loss_fn_deriv):
        for epoch in range(epochs):
            output = self.predict(X)
            loss = loss_fn(y, output)
            error = loss_fn_deriv(y, output)

            # Backward pass
            for layer in reversed(self.layers):
                error = layer.backward(error, learning_rate)

            if epoch % 1000 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch} - Loss: {loss:.5f}")
