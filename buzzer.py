import cv2
import dlib
import numpy as np
import winsound
import time
from scipy.spatial import distance

# ----------------------------
# Function to calculate EAR
# ----------------------------
def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# ----------------------------
# Initialization
# ----------------------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

EAR_THRESHOLD = 0.25
FRAME_LIMIT = 20

counter = 0
blink_count = 0
alarm_on = False

start_time = time.time()

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

        # Draw face bounding box
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        landmarks = predictor(gray, face)

        leftEye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]
        rightEye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]

        leftEAR = calculate_EAR(leftEye)
        rightEAR = calculate_EAR(rightEye)
        EAR = (leftEAR + rightEAR) / 2.0

        # Draw eye landmarks
        for (x, y) in leftEye + rightEye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Display EAR
        cv2.putText(frame, f"EAR: {EAR:.2f}", (450, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # ----------------------------
        # Drowsiness Detection Logic
        # ----------------------------
        if EAR < EAR_THRESHOLD:
            counter += 1

            if counter >= FRAME_LIMIT:
                cv2.putText(frame, "DROWSY ALERT!", (120, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

                if not alarm_on:
                    winsound.Beep(1500, 800)
                    alarm_on = True
        else:
            if counter >= 3:
                blink_count += 1
            counter = 0
            alarm_on = False

        # Status
        status = "Drowsy" if counter >= FRAME_LIMIT else "Awake"
        color = (0, 0, 255) if status == "Drowsy" else (0, 255, 0)

        cv2.putText(frame, f"Status: {status}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.putText(frame, f"Blinks: {blink_count}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # ----------------------------
    # FPS Calculation
    # ----------------------------
    fps = int(1 / (time.time() - start_time))
    start_time = time.time()

    cv2.putText(frame, f"FPS: {fps}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Driver Drowsiness Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
