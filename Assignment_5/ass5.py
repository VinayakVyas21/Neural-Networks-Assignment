import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load Fashion MNIST
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Preprocess: normalize & add channel dimension
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32')  / 255.0
x_train = np.expand_dims(x_train, -1)  # (60000, 28, 28, 1)
x_test  = np.expand_dims(x_test, -1)

# One-hot encode labels
y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
y_test_cat  = tf.keras.utils.to_categorical(y_test, 10)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def build_cnn(filter_size=(3,3), dropout_rate=0.25):
    model = models.Sequential([
        layers.Conv2D(32, filter_size, activation='relu', padding='same', input_shape=(28,28,1)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, filter_size, activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, filter_size, activation='relu', padding='same'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(10, activation='softmax')
    ])
    return model\
# Example configurations to test
configs = [
    {'name': 'Baseline (3x3, Adam, bs=64, dropout=0.25)', 'filter_size': (3,3), 'dropout': 0.25, 'opt': 'adam', 'bs': 64},
    {'name': 'Larger filter (5x5)', 'filter_size': (5,5), 'dropout': 0.25, 'opt': 'adam', 'bs': 64},
    {'name': 'Higher dropout (0.5)', 'filter_size': (3,3), 'dropout': 0.5,  'opt': 'adam', 'bs': 64},
    {'name': 'Small batch (32)', 'filter_size': (3,3), 'dropout': 0.25, 'opt': 'adam', 'bs': 32},
    {'name': 'SGD momentum', 'filter_size': (3,3), 'dropout': 0.25, 'opt': tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9), 'bs': 64},
]

results = []
for cfg in configs:
    print(f"\nTraining: {cfg['name']}")
    model = build_cnn(cfg['filter_size'], cfg['dropout'])
    model.compile(optimizer=cfg['opt'],
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(x_train, y_train_cat,
                        batch_size=cfg['bs'],
                        epochs=15,
                        validation_split=0.2,
                        verbose=1)
    
    test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    
    results.append((cfg, history, test_acc))
    
    # Optional: plot curves
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.plot(history.history['loss'], label='train'); plt.plot(history.history['val_loss'], label='val'); plt.title('Loss'); plt.legend()
    plt.subplot(1,2,2); plt.plot(history.history['accuracy'], label='train'); plt.plot(history.history['val_accuracy'], label='val'); plt.title('Accuracy'); plt.legend()
    plt.suptitle(cfg['name'])
    plt.show()

# Summary
for cfg, hist, acc in results:
    print(f"{cfg['name']}: Test Acc = {acc:.4f} | Best Val Acc = {max(hist.history['val_accuracy']):.4f}")
