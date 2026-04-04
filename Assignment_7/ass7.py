import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# NOTE: Download dataset from https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
# Extract to a folder named 'chest_xray' containing train/, val/, test/ subfolders
base_dir = 'chest_xray'

train_dir = base_dir + '/train'
val_dir   = base_dir + '/val'
test_dir  = base_dir + '/test'

# Data augmentation for training (helps generalization on medical images)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.9, 1.1]
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=32,
    class_mode='binary', color_mode='rgb', shuffle=True
)

val_gen = val_test_datagen.flow_from_directory(
    val_dir, target_size=(224, 224), batch_size=32,
    class_mode='binary', color_mode='rgb', shuffle=False
)

test_gen = val_test_datagen.flow_from_directory(
    test_dir, target_size=(224, 224), batch_size=32,
    class_mode='binary', color_mode='rgb', shuffle=False
)

# Load pre-trained base model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False   # Freeze for transfer learning

# Build model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train
history = model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen,
    verbose=1
)

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_gen, verbose=0)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Predictions for confusion matrix
y_pred_prob = model.predict(test_gen)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()
y_true = test_gen.classes

# Visualizations
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Pneumonia'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix on Test Set')
plt.show()
