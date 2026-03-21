import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import numpy as np
import tensorflow as tf
import pickle
import time

# Initialize MediaPipe Pose Landmarker
base_options = BaseOptions(model_asset_path='pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False)
detector = vision.PoseLandmarker.create_from_options(options)

# Load models and label encoder
# (Assuming models will be saved as .keras in newer TF)
try:
    model_single = tf.keras.models.load_model('model_single_frame.keras')
    model_seq = tf.keras.models.load_model('model_sequence.keras')
except:
    model_single = None
    model_seq = None

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

MAX_SEQ_LENGTH = 30

def analyze_yoga(video_path=0):  # 0 for webcam
    cap = cv2.VideoCapture(video_path)
    
    sequence = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to RGB
        image_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Process
        detection_result = detector.detect(mp_image)
        
        # Display image
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        if detection_result.pose_landmarks:
            # Note: drawing landmarks with Tasks API requires custom code or mp.solutions.drawing_utils
            # Since I removed mp.solutions, I'll just skip drawing for now or use a simple way.
            
            # Extract landmarks for prediction
            landmarks = []
            for lm in detection_result.pose_landmarks[0]:
                landmarks.extend([lm.x, lm.y, lm.z, lm.visibility if hasattr(lm, 'visibility') else 1.0])
            
            # Single frame prediction
            if model_single:
                prediction_single = model_single.predict(np.expand_dims(landmarks, axis=0), verbose=0)
                class_single = le.classes_[np.argmax(prediction_single)]
                prob_single = np.max(prediction_single)
            else:
                class_single = "Model not loaded"
                prob_single = 0
            
            # Sequence prediction
            sequence.append(landmarks)
            sequence = sequence[-MAX_SEQ_LENGTH:]
            
            if len(sequence) == MAX_SEQ_LENGTH and model_seq:
                prediction_seq = model_seq.predict(np.expand_dims(sequence, axis=0), verbose=0)
                class_seq = le.classes_[np.argmax(prediction_seq)]
                prob_seq = np.max(prediction_seq)
            else:
                class_seq = "Gathering data..."
                prob_seq = 0
                
            # Display results on frame
            cv2.putText(image, f"Single Frame: {class_single} ({prob_single:.2f})", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Sequence: {class_seq} ({prob_seq:.2f})", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Suggestion based on quality
            if "poor" in class_single or "poor" in class_seq:
                suggestion = "Focus on alignment and steady breathing."
            elif "avg" in class_single or "avg" in class_seq:
                suggestion = "Good effort! Try to hold the pose longer."
            else:
                suggestion = "Excellent posture! Maintain focus."
            
            cv2.putText(image, f"Feedback: {suggestion}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Show the image
        cv2.imshow('Yoga Posture Analysis', image)
        
        if cv2.waitKey(5) & 0xFF == 27:  # Press ESC to exit
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Check if a video path is provided as an argument
    import sys
    if len(sys.argv) > 1:
        video_to_test = sys.argv[1]
        print(f"Analyzing video: {video_to_test}")
        analyze_yoga(video_to_test)
    else:
        # Default to webcam
        print("Starting webcam analysis...")
        analyze_yoga(0)
