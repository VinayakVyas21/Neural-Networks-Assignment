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

# Constants
MAX_SEQ_LENGTH = 10  # Reduced for more sequences per video

# Load processed data
with open('processed_data.pkl', 'rb') as f:
    data = pickle.load(f)

X = []
y = []

# Prepare sequences: for each video, we can extract multiple sequences
# Each sequence will have MAX_SEQ_LENGTH frames
for item in data:
    asana = item['asana']
    quality = item['quality']
    label = f"{asana}_{quality}"
    
    landmarks_seq = np.array(item['landmarks'])
    
    if len(landmarks_seq) < MAX_SEQ_LENGTH:
        # Pad with zeros if sequence is too short
        pad_size = MAX_SEQ_LENGTH - len(landmarks_seq)
        padded_seq = np.pad(landmarks_seq, ((0, pad_size), (0, 0)), mode='constant')
        X.append(padded_seq)
        y.append(label)
    else:
        # Sliding window with overlap
        for i in range(0, len(landmarks_seq) - MAX_SEQ_LENGTH + 1, 5):
            X.append(landmarks_seq[i:i + MAX_SEQ_LENGTH])
            y.append(label)

X = np.array(X)
y = np.array(y)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build model (LSTM/GRU)
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, input_shape=(MAX_SEQ_LENGTH, X.shape[2]), return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
y_pred = np.argmax(model.predict(X_test), axis=1)

# Metrics
print(f"Model 2: Sequence of Frames Analysis")
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
plt.title('Confusion Matrix - Model 2 (Sequence of Frames)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix_model2.png')

# Save model
model.save('model_sequence.keras')

print("Training complete for Model 2.")
