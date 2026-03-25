import cv2
import dlib
import numpy as np
import gradio as gr
from scipy.spatial import distance
import time

# Load dlib models
try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
except Exception as e:
    print(f"Error loading model: {e}")
    detector = None
    predictor = None

def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def process_stream(frame, threshold, state):
    if frame is None:
        return frame, state, "No Frame Detected"
        
    if detector is None or predictor is None:
        return frame, state, "Model Not Loaded"

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

    # Gradio provided image is already RGB
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = detector(gray)
    
    status = "Awake"
    color = (0, 255, 0) # Green for Awake (RGB)

    for face in faces:
        # Draw face box
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # Blue box in RGB

        landmarks = predictor(gray, face)
        
        leftEye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]
        rightEye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]
        
        leftEAR = calculate_EAR(leftEye)
        rightEAR = calculate_EAR(rightEye)
        EAR = (leftEAR + rightEAR) / 2.0
        
        # Draw eyes
        for pt in leftEye + rightEye:
            cv2.circle(frame, pt, 2, (0, 255, 0), -1)

        # Alert logic
        if EAR < threshold:
            counter += 1
            if counter >= 10: # Reduced for faster response in web app
                status = "Drowsy"
                color = (255, 0, 0) # Red status (RGB)
                cv2.putText(frame, "!! DROWSY !!", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        else:
            if counter >= 3: # Blink detection
                blink_count += 1
            counter = 0

        cv2.putText(frame, f"EAR: {EAR:.2f}", (x+w-80, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    state["counter"] = counter
    state["blink_count"] = blink_count

    # UI Overlays
    cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"Blinks: {blink_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1]-100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    return frame, state, f"Current Status: {status} | Total Blinks: {blink_count}"

# Define custom CSS for a premium look
custom_css = """
.container { max-width: 1000px; margin: auto; }
footer { visibility: hidden; }
"""

with gr.Blocks(css=custom_css, title="Driver Drowsiness Detector") as demo:
    gr.Markdown("<h1 style='text-align: center;'>🚗 Driver Drowsiness Monitoring System</h1>")
    gr.Markdown("<p style='text-align: center;'>AI-powered real-time detection for safer driving. Use the webcam feed below.</p>")
    
    app_state = gr.State({"counter": 0, "blink_count": 0, "last_time": time.time()})
    
    with gr.Row():
        with gr.Column(scale=2):
            # webcam input with streaming enabled
            input_img = gr.Image(sources=["webcam"], label="Webcam Feed", type="numpy", streaming=True)
            threshold = gr.Slider(0.15, 0.35, value=0.25, step=0.01, label="Detection Sensitivity (EAR Threshold)")
        
        with gr.Column(scale=2):
            output_img = gr.Image(label="Live Analytics")
            status_text = gr.Textbox(label="Real-time Metrics")

    # Connect the stream for continuous processing
    input_img.stream(
        fn=process_stream,
        inputs=[input_img, threshold, app_state],
        outputs=[output_img, app_state, status_text],
        stream_every=0.05 # Process every 50ms for a "live" feel
    )

if __name__ == "__main__":
    demo.launch()
