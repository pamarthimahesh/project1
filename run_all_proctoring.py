import cv2
import os
import time
from datetime import datetime
from multiple_face_eye_tracking import detect_face_and_direction
from record_audio import record_audio_clip
from predict_emotion import predict_emotion_from_audio

# Create base folders
os.makedirs("violations", exist_ok=True)
os.makedirs("predictions", exist_ok=True)

# Cooldown dictionary to prevent repeated saves for same violation
violation_cooldown = {}

# Save snapshot
def save_snapshot(frame, violation_type):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join("violations", violation_type)
    os.makedirs(folder, exist_ok=True)
    filename = f"{violation_type}_{timestamp}.jpg"
    filepath = os.path.join(folder, filename)
    cv2.imwrite(filepath, frame)
    print(f"[INFO] Snapshot saved: {filepath}")
    return filepath

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot access webcam.")
        return

    print("[INFO] Webcam access successful. Starting proctoring...")
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame not captured.")
            break

        result = detect_face_and_direction(frame)
        status_text = result["status"]

        if result["violation"]:
            violation_type = result["violation_type"]
            current_time = time.time()

            # Check cooldown to avoid repeated violation logs
            last_time = violation_cooldown.get(violation_type, 0)
            if current_time - last_time > 5:  # 5 seconds cooldown
                print(f"[VIOLATION DETECTED] {violation_type} at {datetime.now().strftime('%H:%M:%S')}")

                # Save snapshot
                save_snapshot(frame, violation_type)

                # Record audio
                audio_path = record_audio_clip(duration=5, violation_type=violation_type)

                # Predict emotion
                if audio_path:
                    emotion = predict_emotion_from_audio(audio_path)
                    if emotion:
                        pred_filename = f"{emotion}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                        pred_path = os.path.join("predictions", pred_filename)
                        os.rename(audio_path, pred_path)
                        print(f"[INFO] Emotion audio saved to: {pred_path}")

                # Update cooldown timestamp
                violation_cooldown[violation_type] = current_time

        # Show status on frame
        cv2.putText(frame, f"Status: {status_text}", (10, 30), font, 0.7, (0, 0, 255), 2)
        cv2.imshow("Mock Interview Proctoring", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
