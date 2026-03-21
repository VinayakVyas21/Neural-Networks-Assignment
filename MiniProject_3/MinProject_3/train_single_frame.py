import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import seaborn as sns
import os

# Load processed data
with open('processed_data.pkl', 'rb') as f:
    data = pickle.load(f)

X = []
y = []

# For single frame model, we treat each frame as an independent sample
# The target is the 'asana' (pose type). The quality is secondary for now, 
# but the prompt asks for "analyzing and suggesting improvements", 
# which implies we should classify the quality (good/avg/poor).
# Let's create a combined label: 'asana_quality'
for item in data:
    asana = item['asana']
    quality = item['quality']
    label = f"{asana}_{quality}"
    for frame in item['landmarks']:
        X.append(frame)
        y.append(label)

X = np.array(X)
y = np.array(y)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
y_pred = np.argmax(model.predict(X_test), axis=1)

# Metrics
print(f"Model 1: Single Frame Analysis")
print(f"Accuracy: {accuracy:.4f}")
print(f"Loss: {loss:.4f}")

# Get unique classes in the test set to avoid error
unique_test_classes = np.unique(y_test)
target_names = [le.classes_[i] for i in unique_test_classes]

print("Classification Report:")
print(classification_report(y_test, y_pred, labels=unique_test_classes, target_names=target_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix - Model 1 (Single Frame)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix_model1.png')

# Save model and encoder
model.save('model_single_frame.keras')
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("Training complete for Model 1.")
