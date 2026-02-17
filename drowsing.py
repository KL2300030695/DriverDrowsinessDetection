import cv2
import dlib
import numpy as np
import winsound
from scipy.spatial import distance

# ----------------------------
# Function to calculate EAR
# ----------------------------
def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    EAR = (A + B) / (2.0 * C)
    return EAR

# ----------------------------
# Initialization
# ----------------------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

# Drowsiness parameters
EAR_THRESHOLD = 0.25     # Try between 0.23 - 0.30
FRAME_LIMIT = 20         # Number of consecutive frames
counter = 0

# ----------------------------
# Main Loop
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        leftEye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]
        rightEye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]

        leftEAR = calculate_EAR(leftEye)
        rightEAR = calculate_EAR(rightEye)
        EAR = (leftEAR + rightEAR) / 2.0

        # Draw eye landmarks
        for (x, y) in leftEye + rightEye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Display EAR value
        cv2.putText(frame, f"EAR: {EAR:.2f}", (450, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Drowsiness logic
        if EAR < EAR_THRESHOLD:
            counter += 1
            if counter >= FRAME_LIMIT:
                cv2.putText(frame, "DROWSY ALERT!", (150, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        else:
            counter = 0

        # Status display
        status = "Drowsy" if counter >= FRAME_LIMIT else "Awake"
        color = (0, 0, 255) if status == "Drowsy" else (0, 255, 0)

        cv2.putText(frame, f"Status: {status}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Driver Drowsiness Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
