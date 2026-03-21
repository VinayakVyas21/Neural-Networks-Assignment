import pickle
import os

data = []
with open('processed_data_new.pkl', 'rb') as f:
    while True:
        try:
            data.append(pickle.load(f))
        except EOFError:
            break

with open('processed_data.pkl', 'wb') as f:
    pickle.dump(data, f)
print(f"Consolidated {len(data)} videos into processed_data.pkl")
