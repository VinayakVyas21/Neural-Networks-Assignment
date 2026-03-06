import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

# Load and preprocess MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784) / 255.0
x_test  = x_test.reshape(-1, 784) / 255.0
y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
y_test_cat  = tf.keras.utils.to_categorical(y_test, 10)

def build_and_train(hidden_size=128, activation='relu',
                    learning_rate=0.001, batch_size=32, epochs=15):
    model = models.Sequential([
        layers.Dense(hidden_size, activation=activation, input_shape=(784,)),
        layers.Dense(hidden_size // 2, activation=activation),
        layers.Dense(10, activation='softmax')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(x_train, y_train_cat,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.2,
                        verbose=0)
    
    test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
    
    return history, test_acc

# Example experiments – run each combination separately or in loops
configurations = [
    # Baseline
    {'hidden_size': 128, 'activation': 'relu', 'lr': 0.001, 'batch': 32, 'epochs': 15},
    
    # Activation functions
    {'hidden_size': 128, 'activation': 'sigmoid', 'lr': 0.001, 'batch': 32, 'epochs': 30},
    {'hidden_size': 128, 'activation': 'tanh',    'lr': 0.001, 'batch': 32, 'epochs': 20},
    
    # Hidden layer sizes
    {'hidden_size': 64,  'activation': 'relu', 'lr': 0.001, 'batch': 32, 'epochs': 15},
    {'hidden_size': 256, 'activation': 'relu', 'lr': 0.001, 'batch': 32, 'epochs': 15},
    {'hidden_size': 512, 'activation': 'relu', 'lr': 0.001, 'batch': 32, 'epochs': 15},
    
    # Learning rates
    {'hidden_size': 128, 'activation': 'relu', 'lr': 0.01,  'batch': 32, 'epochs': 10},
    {'hidden_size': 128, 'activation': 'relu', 'lr': 0.0001,'batch': 32, 'epochs': 30},
    
    # Batch sizes
    {'hidden_size': 128, 'activation': 'relu', 'lr': 0.001, 'batch': 8,  'epochs': 15},
    {'hidden_size': 128, 'activation': 'relu', 'lr': 0.001, 'batch': 128,'epochs': 15},
]

results = []
for config in configurations:
    print(f"Running: {config}")
    history, test_acc = build_and_train(**config)
    results.append((config, history.history['val_accuracy'][-1], test_acc))
    # Optional: plot loss/accuracy curves here for each run

# Summary table (print or display)
print("\nSummary of Results:")
for conf, val_acc, test_acc in results:
    print(f"Config: {conf} → Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")
