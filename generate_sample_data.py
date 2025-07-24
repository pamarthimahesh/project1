# generate_sample_data.py
import os
from gtts import gTTS

# Define text for each emotion
emotion_data = {
    "happy": [
        "I am feeling great today!",
        "Life is beautiful and exciting."
    ],
    "sad": [
        "I am feeling down and tired.",
        "It's a gloomy and lonely day."
    ],
    "angry": [
        "Why did this happen again?",
        "I am so frustrated with this mess!"
    ]
}

# Output directory
base_dir = "voice_data"

# Generate audio files
for emotion, sentences in emotion_data.items():
    emotion_dir = os.path.join(base_dir, emotion)
    os.makedirs(emotion_dir, exist_ok=True)
    for idx, sentence in enumerate(sentences):
        file_path = os.path.join(emotion_dir, f"{emotion}{idx + 1}.wav")
        tts = gTTS(text=sentence, lang='en')
        tts.save(file_path.replace('.wav', '.mp3'))  # Save as .mp3
        os.system(f"ffmpeg -y -loglevel panic -i {file_path.replace('.wav', '.mp3')} {file_path}")  # Convert to .wav
        os.remove(file_path.replace('.wav', '.mp3'))  # Remove temp .mp3
        print(f"[SAVED] {file_path}")
