import os
import subprocess
from tqdm import tqdm

DATASET_PATH = 'Dataset'
OUTPUT_FILE = 'processed_data_new.pkl'
PYTHON_PATH = os.path.join('yoga_venv', 'Scripts', 'python.exe')

if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

asanas = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
qualities = ['avg', 'good', 'poor']

for asana in tqdm(asanas, desc="Asanas"):
    asana_path = os.path.join(DATASET_PATH, asana)
    for quality in qualities:
        quality_path = os.path.join(asana_path, quality)
        if not os.path.isdir(quality_path):
            continue
            
        videos = [f for f in os.listdir(quality_path) if f.endswith(('.mp4', '.avi', '.mov'))]
        # Process 1 video per quality for speed
        for video in videos[:1]:
            video_path = os.path.join(quality_path, video)
            print(f"Processing {asana} - {quality} - {video}")
            try:
                subprocess.run([PYTHON_PATH, 'process_single_video.py', video_path, asana, quality, OUTPUT_FILE], check=True)
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
