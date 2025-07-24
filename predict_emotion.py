import pickle
import librosa
import numpy as np
import os

MODEL_PATH = "emotion_model.pkl"

def extract_mfcc_features(audio_path, max_len=180):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=180)
        mfccs_mean = np.mean(mfccs, axis=1)
        if len(mfccs_mean) < max_len:
            pad_width = max_len - len(mfccs_mean)
            mfccs_mean = np.pad(mfccs_mean, (0, pad_width))
        elif len(mfccs_mean) > max_len:
            mfccs_mean = mfccs_mean[:max_len]
        return mfccs_mean.reshape(1, -1)
    except Exception as e:
        print(f"[ERROR] Feature extraction failed: {e}")
        return None

def predict_emotion_from_audio(audio_path):
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"[ERROR] Model file not found: {MODEL_PATH}")
            return None

        with open(MODEL_PATH, "rb") as file:
            model = pickle.load(file)

        features = extract_mfcc_features(audio_path)
        if features is None:
            return None

        prediction = model.predict(features)
        confidence = model.predict_proba(features).max()
        print(f"[INFO] Predicted Emotion: {prediction[0]} (Confidence: {confidence:.2f})")
        return prediction[0]
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return None
