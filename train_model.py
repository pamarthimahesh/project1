import os
import pickle
import numpy as np
from extract_features import extract_features_from_audio
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Get absolute path to the dataset folder
DATA_DIR = os.path.abspath("voice_data")

# Lists to hold features and labels
X = []
y = []

# Walk through each subfolder (assumed to be emotion labels)
for emotion_label in os.listdir(DATA_DIR):
    emotion_dir = os.path.join(DATA_DIR, emotion_label)

    # Skip if not a directory
    if not os.path.isdir(emotion_dir):
        continue

    # Loop through each .wav file in the emotion-labeled folder
    for file in os.listdir(emotion_dir):
        if file.endswith(".wav"):
            try:
                file_path = os.path.join(emotion_dir, file)
                print(f"[INFO] Processing: {file_path}")

                # Extract features from the audio file
                features = extract_features_from_audio(file_path)

                # Check if features are valid
                if features is not None and len(features) > 0:
                    X.append(features)
                    y.append(emotion_label)
                else:
                    print(f"[WARNING] Empty features from {file_path}")

            except Exception as e:
                print(f"[WARNING] Skipped {file}: {e}")

# Convert lists to NumPy arrays
if len(X) == 0 or len(y) == 0:
    print("[ERROR] No valid data found. Please check your input files and feature extraction.")
    exit()

# Convert to NumPy arrays with float dtype
X = np.array(X, dtype=np.float32)
y = np.array(y)

# Encode emotion labels (string to numeric)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Debug prints
print(f"[INFO] Sample feature shape: {X[0].shape if len(X) > 0 else 'N/A'}")
print(f"[INFO] Training data shape: {X.shape}")
print(f"[INFO] Labels shape: {y_encoded.shape}")
print(f"[INFO] Emotion classes: {label_encoder.classes_}")

# Train the RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y_encoded)

# Save the trained model
with open("emotion_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save the label encoder for future decoding
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("[INFO] Model and label encoder saved successfully.")
