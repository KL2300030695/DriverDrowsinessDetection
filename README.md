---
title: Driver Drowsiness Detection
emoji: 🚗
colorFrom: blue
colorTo: red
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
python_version: "3.10"
---

# 🚗 Driver Drowsiness Monitoring System

An AI-powered real-time Driver Drowsiness Detection System that monitors a driver’s eye movements using computer vision and alerts them when signs of drowsiness are detected.

## 📌 Overview

Driver fatigue is one of the major causes of road accidents worldwide. This project uses Computer Vision and Facial Landmark detection to monitor eye closure and alert the driver.

The system continuously monitors the driver's face through a webcam and identifies potential drowsiness using the Eye Aspect Ratio (EAR).

## 🎯 Features

* **👁 Real-time eye detection:** Utilizes webcam feed for continuous monitoring.
* **🧠 Facial Landmark Analysis:** Accurate detection using MediaPipe's high-fidelity mesh model.
* **🔔 Live Analytics:** Displays EAR, blink count, and status on an interactive dashboard.
* **⚡ Fast and lightweight:** Optimized for real-time web inference using MediaPipe.
* **🖥 Easy Cloud/Local setup:** Run it as a Hugging Face Space or locally with Gradio.

## 🛠️ Tech Stack

* **Python**
* **OpenCV**
* **MediaPipe** (Facial Landmarks)
* **Gradio** (Web Interface)
* **SciPy** (EAR Calculation)

## 🏗️ Project Structure

```text
DriverDrowsinessDetection/
│
├── app.py                                # Main Gradio application
├── requirements.txt                      # Dependencies
└── README.md
```

## 🚀 Installation & Setup

### 1️⃣ Run as Hugging Face Space
* Create a new Space on [Hugging Face](https://huggingface.co/new-space).
* Select **Gradio** as the SDK.
* Push this repository to the Space.

### 2️⃣ Local Setup
```bash
# Clone the repository
git clone https://github.com/KL2300030695/DriverDrowsinessDetection.git
cd DriverDrowsinessDetection

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

## 👨‍💻 Author
**Subhash Vadaparthi**
B.Tech Computer Science Engineering
Koneru Lakshmaiah Education Foundation
AI & Cloud Enthusiast
