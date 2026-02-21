import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")


class Perceptron:
    """
    Single-layer perceptron with:
    - 2 input neurons
    - 1 bias
    - Step activation function
    """

    def __init__(self, learning_rate=0.1):
        self.weights = np.random.randn(2)
        self.bias = np.random.randn()
        self.lr = learning_rate

    def activation_function(self, x):
        """Step activation function"""
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        """Compute perceptron output"""
        linear_output = np.dot(inputs, self.weights) + self.bias
        return self.activation_function(linear_output)

    def train(self, X, y, epochs=10, title="", save_dir=None):
        """
        Train perceptron and save decision boundary image per epoch
        """

        # Create directory for images
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        plt.figure(figsize=(6, 6))

        for epoch in range(epochs):
            total_error = 0

            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                error = target - prediction

                # Error-correction learning rule
                self.weights += self.lr * error * xi
                self.bias += self.lr * error
                total_error += abs(error)

            # Plot decision boundary
            self._plot_decision_boundary(X, y, epoch, title)

            # Save image for this epoch
            if save_dir:
                filename = f"{save_dir}/{title.replace(' ', '_')}_epoch_{epoch}.png"
                plt.savefig(filename, dpi=300)

            # Stop early if converged
            if total_error == 0:
                print(f"{title} converged at epoch {epoch}")
                break

            plt.pause(0.5)

        plt.show()

    def _plot_decision_boundary(self, X, y, epoch, title):
        """Plot data points and decision boundary"""
        plt.clf()

        # Plot input points
        for i in range(len(X)):
            color = "green" if y[i] == 1 else "red"
            plt.scatter(X[i][0], X[i][1], color=color, s=100)

        # Decision boundary: w1*x1 + w2*x2 + b = 0
        x_vals = np.array([-0.5, 1.5])
        if self.weights[1] != 0:
            y_vals = -(self.weights[0] * x_vals + self.bias) / self.weights[1]
            plt.plot(x_vals, y_vals, 'b--', linewidth=2)

        plt.xlim(-0.5, 1.5)
        plt.ylim(-0.5, 1.5)
        plt.title(f"{title} | Epoch {epoch}")


# MAIN PROGRAM
if __name__ == "__main__":

    # NAND Dataset 
    nand_inputs = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    nand_labels = np.array([1, 1, 1, 0])

    print("Training on NAND dataset")
    perceptron_nand = Perceptron(learning_rate=0.1)
    perceptron_nand.train(
        nand_inputs,
        nand_labels,
        epochs=10,
        title="NAND Perceptron",
        save_dir="nand_images"
    )

    # XOR Dataset 
    xor_inputs = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    xor_labels = np.array([0, 1, 1, 0])

    print("\nTraining on XOR dataset (expected to fail)")
    perceptron_xor = Perceptron(learning_rate=0.1)
    perceptron_xor.train(
        xor_inputs,
        xor_labels,
        epochs=10,
        title="XOR Perceptron",
        save_dir="xor_images"
    )
