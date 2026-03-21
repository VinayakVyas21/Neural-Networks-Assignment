import pickle
import os

data = []
if os.path.exists('processed_data_new.pkl'):
    with open('processed_data_new.pkl', 'rb') as f:
        while True:
            try:
                data.append(pickle.load(f))
            except EOFError:
                break
    print(f"Loaded {len(data)} videos")
    asanas = sorted(list(set(item['asana'] for item in data)))
    print(f"Asanas: {asanas}")
    for asana in asanas:
        qualities = sorted(list(set(item['quality'] for item in data if item['asana'] == asana)))
        print(f"  {asana}: {qualities}")
else:
    print("File not found.")
