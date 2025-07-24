import librosa
import numpy as np

def extract_features_from_audio(file_path, max_pad_len=180):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')  # load audio file
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=180)
        mfccs_processed = np.mean(mfccs.T, axis=0)

        # If fewer than 180, pad it
        if len(mfccs_processed) < 180:
            pad_width = 180 - len(mfccs_processed)
            mfccs_processed = np.pad(mfccs_processed, (0, pad_width), mode='constant')
        else:
            mfccs_processed = mfccs_processed[:180]

        return mfccs_processed
    except Exception as e:
        print(f"[ERROR] Feature extraction failed: {e}")
        return np.zeros(180)
