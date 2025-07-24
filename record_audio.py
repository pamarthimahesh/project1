import sounddevice as sd
import scipy.io.wavfile as wav
import os
from datetime import datetime

def record_audio_clip(duration=5, violation_type="unknown"):
    try:
        fs = 44100
        print("[INFO] Recording audio...")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        print("[INFO] Recording complete.")

        folder = os.path.join("violations", violation_type)
        os.makedirs(folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{violation_type}_{timestamp}.wav"
        filepath = os.path.join(folder, filename)
        wav.write(filepath, fs, recording)
        print(f"[INFO] Saved to {filepath}")
        return filepath
    except Exception as e:
        print(f"[ERROR] Recording failed: {e}")
        return None
