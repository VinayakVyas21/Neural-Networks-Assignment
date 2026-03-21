import os
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import numpy as np
import pickle
from tqdm import tqdm
import sys

# Log to file
log_file = open("preprocess_log.txt", "w", encoding='utf-8')
sys.stdout = log_file
sys.stderr = log_file

print("Initializing Pose Landmarker...", flush=True)
try:
    base_options = BaseOptions(model_asset_path='pose_landmarker.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False)
    detector = vision.PoseLandmarker.create_from_options(options)
    print("Pose Landmarker initialized.", flush=True)
except Exception as e:
    print(f"Error initializing detector: {e}", flush=True)
    sys.exit(1)

DATASET_PATH = 'Dataset'
OUTPUT_FILE = 'processed_data.pkl'

asanas = os.listdir(DATASET_PATH)
qualities = ['avg', 'good', 'poor']

data = []

def extract_landmarks(video_path):
    cap = cv2.VideoCapture(video_path)
    landmarks_seq = []
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 2 != 0: # Process every 2nd frame
            continue
            
        # Convert to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        
        # Process
        detection_result = detector.detect(mp_image)
        
        if detection_result.pose_landmarks:
            # We take the first pose detected
            landmarks = []
            for lm in detection_result.pose_landmarks[0]:
                landmarks.extend([lm.x, lm.y, lm.z, lm.visibility if hasattr(lm, 'visibility') else 1.0])
            landmarks_seq.append(landmarks)
        
    cap.release()
    return landmarks_seq

print("Starting landmark extraction...", flush=True)
print(f"Asanas found: {asanas}", flush=True)

for asana in tqdm(asanas, desc="Asanas"):
    asana_path = os.path.join(DATASET_PATH, asana)
    if not os.path.isdir(asana_path):
        print(f"Skipping {asana}, not a directory", flush=True)
        continue
    
    print(f"Processing asana: {asana}", flush=True)
    for quality in qualities:
        quality_path = os.path.join(asana_path, quality)
        if not os.path.isdir(quality_path):
            print(f"Skipping {quality} for {asana}, not a directory", flush=True)
            continue
            
        print(f"Processing {quality} for {asana}", flush=True)
        videos = [f for f in os.listdir(quality_path) if f.endswith(('.mp4', '.avi', '.mov'))]
        print(f"Videos found: {len(videos)}", flush=True)
        # Process at most 1 video per category for speed
        for video in tqdm(videos[:1], desc=f"{asana} - {quality}", leave=False):
            video_path = os.path.join(quality_path, video)
            print(f"Extracting from: {video_path}", flush=True)
            try:
                landmarks_seq = extract_landmarks(video_path)
                print(f"Extracted {len(landmarks_seq)} frames", flush=True)
                
                if landmarks_seq:
                    data.append({
                        'asana': asana,
                        'quality': quality,
                        'landmarks': landmarks_seq
                    })
                
                # Save partial data
                with open(OUTPUT_FILE, 'wb') as f:
                    pickle.dump(data, f)
            except Exception as e:
                print(f"Error processing {video_path}: {e}")

# Save data
with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(data, f)

print(f"Extraction complete. Data saved to {OUTPUT_FILE}")
log_file.close()
