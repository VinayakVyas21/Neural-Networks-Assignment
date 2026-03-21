import os
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import numpy as np
import pickle
import sys

# Initialize MediaPipe Pose Landmarker
try:
    base_options = BaseOptions(model_asset_path='pose_landmarker.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False)
    detector = vision.PoseLandmarker.create_from_options(options)
except Exception as e:
    sys.exit(1)

def extract_landmarks(video_path):
    cap = cv2.VideoCapture(video_path)
    landmarks_seq = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 3 != 0: # Process every 3rd frame for even more speed
            continue
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        detection_result = detector.detect(mp_image)
        if detection_result.pose_landmarks:
            landmarks = []
            for lm in detection_result.pose_landmarks[0]:
                landmarks.extend([lm.x, lm.y, lm.z, lm.visibility if hasattr(lm, 'visibility') else 1.0])
            landmarks_seq.append(landmarks)
    cap.release()
    return landmarks_seq

if __name__ == "__main__":
    video_path = sys.argv[1]
    asana = sys.argv[2]
    quality = sys.argv[3]
    output_file = sys.argv[4]
    
    try:
        landmarks = extract_landmarks(video_path)
        if landmarks:
            result = {
                'asana': asana,
                'quality': quality,
                'landmarks': landmarks
            }
            # Append to file
            with open(output_file, 'ab') as f:
                pickle.dump(result, f)
            print(f"Success: {video_path}")
    except Exception as e:
        print(f"Error: {video_path} - {e}")
