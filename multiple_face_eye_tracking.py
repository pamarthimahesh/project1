import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

def get_head_pose(landmarks, frame_shape):
    image_points = np.array([
        landmarks[1],   # Nose tip
        landmarks[152], # Chin
        landmarks[263], # Right eye right corner
        landmarks[33],  # Left eye left corner
        landmarks[287], # Right mouth corner
        landmarks[57],  # Left mouth corner
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (150.0, -150.0, -125.0),     # Right mouth corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
    ])

    height, width = frame_shape
    focal_length = width
    center = (width / 2, height / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

    pitch, yaw, roll = [float(angle) for angle in euler_angles]
    return pitch, yaw, roll

def detect_face_and_direction(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if not results.multi_face_landmarks:
        return {"status": "No face detected", "violation": True, "violation_type": "no_face"}

    if len(results.multi_face_landmarks) > 1:
        return {"status": "Multiple faces detected", "violation": True, "violation_type": "multiple_faces"}

    landmarks = results.multi_face_landmarks[0]
    h, w, _ = frame.shape
    landmark_points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks.landmark]

    pitch, yaw, roll = get_head_pose(landmark_points, (h, w))

    if abs(yaw) > 20:
        return {"status": "Looking away", "violation": True, "violation_type": "looking_away"}

    return {"status": "Face detected", "violation": False, "violation_type": None}

# Optional Webcam Loop
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detect_face_and_direction(frame)
        cv2.putText(frame, result["status"], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if not result["violation"] else (0, 0, 255), 2)

        cv2.imshow("Face Direction Monitor", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
