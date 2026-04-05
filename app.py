import cv2
print("--- Application Starting ---")
import mediapipe as mp
import numpy as np
import gradio as gr
from scipy.spatial import distance
import time
import os

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmark indices for EAR
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def calculate_EAR(eye_landmarks):
    # eye_landmarks: list of (x, y) tuples
    # Vertical distances
    A = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
    # Horizontal distance
    C = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear

def process_stream(frame, threshold, state):
    if frame is None:
        return frame, state, "No Frame Detected"
        
    # Ensure state is a dictionary
    if not isinstance(state, dict):
        state = {"counter": 0, "blink_count": 0, "last_time": time.time()}

    counter = state.get("counter", 0)
    blink_count = state.get("blink_count", 0)
    last_time = state.get("last_time", time.time())
    
    # Calculate FPS
    current_time = time.time()
    dt = current_time - last_time
    fps = 1.0 / dt if dt > 0 else 0
    state["last_time"] = current_time

    # Process frame with MediaPipe
    # Gradio provided image is already RGB
    results = face_mesh.process(frame)
    
    status = "Awake"
    color = (0, 255, 0) # Green for Awake (RGB)
    h, w, _ = frame.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract landmark coordinates
            coords = []
            for lm in face_landmarks.landmark:
                coords.append((int(lm.x * w), int(lm.y * h)))

            # EAR calculation
            left_eye_pts = [coords[i] for i in LEFT_EYE]
            right_eye_pts = [coords[i] for i in RIGHT_EYE]
            
            left_ear = calculate_EAR(left_eye_pts)
            right_ear = calculate_EAR(right_eye_pts)
            EAR = (left_ear + right_ear) / 2.0
            
            # Draw eyes
            for pt in left_eye_pts + right_eye_pts:
                cv2.circle(frame, pt, 1, (0, 255, 0), -1)

            # Alert logic
            if EAR < threshold:
                counter += 1
                if counter >= 10: 
                    status = "Drowsy"
                    color = (255, 0, 0) # Red
                    cv2.putText(frame, "!! DROWSY !!", (coords[10][0]-50, coords[10][1]-20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            else:
                if counter >= 3: # Blink detection
                    blink_count += 1
                counter = 0

            cv2.putText(frame, f"EAR: {EAR:.2f}", (int(coords[33][0]), int(coords[33][1]-30)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    state["counter"] = counter
    state["blink_count"] = blink_count

    # UI Overlays
    cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"Blinks: {blink_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1]-110, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    return frame, state, f"Current Status: {status} | Total Blinks: {blink_count}"

# Define custom CSS for a premium look
custom_css = """
.container { max-width: 1000px; margin: auto; }
footer { visibility: hidden; }
"""

with gr.Blocks(css=custom_css, title="Driver Drowsiness Detector") as demo:
    gr.Markdown("<h1 style='text-align: center;'>🚗 Driver Drowsiness Monitoring System</h1>")
    gr.Markdown("<p style='text-align: center;'>AI-powered real-time detection for safer driving. Powered by MediaPipe.</p>")
    
    app_state = gr.State({"counter": 0, "blink_count": 0, "last_time": time.time()})
    
    with gr.Row():
        with gr.Column(scale=2):
            input_img = gr.Image(sources=["webcam"], label="Webcam Feed", type="numpy", streaming=True)
            threshold = gr.Slider(0.15, 0.35, value=0.25, step=0.01, label="Detection Sensitivity (EAR Threshold)")
        
        with gr.Column(scale=2):
            output_img = gr.Image(label="Live Analytics")
            status_text = gr.Textbox(label="Real-time Metrics")

    # Connect the stream
    input_img.stream(
        fn=process_stream,
        inputs=[input_img, threshold, app_state],
        outputs=[output_img, app_state, status_text],
        stream_every=0.05
    )

if __name__ == "__main__":
    demo.launch()
