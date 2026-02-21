import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------
# CREATE OUTPUT FOLDER
# -----------------------------
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# MODIFIED XOR DATASET
# -----------------------------
X = np.array([
    [0.2, 0.1],
    [0.2, 0.9],
    [0.8, 0.1],
    [0.8, 0.9]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

# -----------------------------
# NETWORK PARAMETERS
# -----------------------------
np.random.seed(42)

input_size = 2
hidden_size = 4
output_size = 1

learning_rate = 0.1
epochs = 10000

n_samples = X.shape[0]

# Xavier Initialization
W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1/input_size)
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1/hidden_size)
b2 = np.zeros((1, output_size))

# -----------------------------
# ACTIVATION FUNCTIONS
# -----------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(a):
    return a * (1 - a)

# -----------------------------
# TRAINING
# -----------------------------
loss_list = []
accuracy_list = []

for epoch in range(epochs):

    # Forward propagation
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W2) + b2
    y_pred = sigmoid(final_input)

    # Loss calculation
    loss = np.mean((y - y_pred) ** 2)
    loss_list.append(loss)

    # Accuracy calculation
    predictions = (y_pred >= 0.5).astype(int)
    accuracy = np.mean(predictions == y)
    accuracy_list.append(accuracy)

    # Backpropagation
    error_output = (y_pred - y) * sigmoid_derivative(y_pred)

    dW2 = (1/n_samples) * np.dot(hidden_output.T, error_output)
    db2 = (1/n_samples) * np.sum(error_output, axis=0, keepdims=True)

    error_hidden = np.dot(error_output, W2.T) * sigmoid_derivative(hidden_output)

    dW1 = (1/n_samples) * np.dot(X.T, error_hidden)
    db1 = (1/n_samples) * np.sum(error_hidden, axis=0, keepdims=True)

    # Weight updates
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    # Print every 500 epochs only
    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1} | Loss: {loss:.6f} | Accuracy: {accuracy:.2f}")

# -----------------------------
# FINAL RESULTS
# -----------------------------
print("\nFinal Predictions:")
print(y_pred)

print("\nRounded Predictions:")
print((y_pred >= 0.5).astype(int))

print("\nFinal Loss:", loss)

print("\nFinal Accuracy:", accuracy)

# -----------------------------
# SAVE LOSS GRAPH
# -----------------------------
plt.figure()
plt.plot(loss_list)
plt.title("Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.savefig(os.path.join(output_dir, "loss_vs_epoch.png"))
plt.close()

# -----------------------------
# SAVE ACCURACY GRAPH
# -----------------------------
plt.figure()
plt.plot(accuracy_list)
plt.title("Accuracy vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid()
plt.savefig(os.path.join(output_dir, "accuracy_vs_epoch.png"))
plt.close()

# -----------------------------
# DECISION BOUNDARY
# -----------------------------
xx, yy = np.meshgrid(np.linspace(0, 1, 200),
                     np.linspace(0, 1, 200))

grid = np.c_[xx.ravel(), yy.ravel()]

hidden_layer = sigmoid(np.dot(grid, W1) + b1)
output_layer = sigmoid(np.dot(hidden_layer, W2) + b2)

Z = output_layer.reshape(xx.shape)

plt.figure()

plt.contourf(xx, yy, Z >= 0.5, alpha=0.3)

plt.scatter(X[y.flatten()==0][:,0],
            X[y.flatten()==0][:,1],
            marker='o',
            s=100,
            label="Class 0")

plt.scatter(X[y.flatten()==1][:,0],
            X[y.flatten()==1][:,1],
            marker='x',
            s=100,
            label="Class 1")

plt.title("Decision Boundary (Modified XOR)")
plt.xlabel("Input 1")
plt.ylabel("Input 2")
plt.legend()

plt.savefig(os.path.join(output_dir, "decision_boundary.png"))
plt.close()

print("\nAll plots saved in 'output' folder.")