import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic sine-wave time series (with slight noise)
np.random.seed(42)
torch.manual_seed(42)
time = np.arange(0, 400)
data = np.sin(time * 0.1) + 0.05 * np.random.randn(len(time))

# Create sequences: predict next value from previous 20 steps
seq_length = 20
X, y = [], []
for i in range(len(data) - seq_length):
    X.append(data[i:i+seq_length])
    y.append(data[i+seq_length])
X = np.array(X).reshape(-1, seq_length, 1)   # (samples, seq_len, features)
y = np.array(y).reshape(-1, 1)

# Convert to PyTorch tensors
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).float()

# Split: 80% train, 20% test
train_size = int(0.8 * len(X))
X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

# Simple RNN model
class SimpleRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=32, num_layers=1, batch_first=True)
        self.fc = nn.Linear(32, 1)
    
    def forward(self, x):
        out, _ = self.rnn(x)          # out shape: (batch, seq_len, hidden)
        out = self.fc(out[:, -1, :])  # take only the last time step
        return out

model = SimpleRNN()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
losses = []
epochs = 150
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()                    # Backpropagation through time
    optimizer.step()
    
    losses.append(loss.item())
    if epoch % 30 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f}")

# Evaluation
model.eval()
with torch.no_grad():
    y_pred_train = model(X_train).numpy()
    y_pred_test  = model(X_test).numpy()

# Plot results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(losses, color='teal')
plt.title('Training Loss Curve (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(time[seq_length:seq_length+len(y_train)], y_train.numpy(), label='Train Actual', color='blue')
plt.plot(time[seq_length:seq_length+len(y_train)], y_pred_train, label='Train Predicted', color='red', linestyle='--')
plt.plot(time[seq_length+len(y_train):], y_test.numpy(), label='Test Actual', color='green')
plt.plot(time[seq_length+len(y_train):], y_pred_test, label='Test Predicted', color='orange', linestyle='--')
plt.title('Actual vs Predicted Time Series')
plt.xlabel('Time step')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
