# 🚗 Driver Drowsiness Detection System

An AI-powered real-time Driver Drowsiness Detection System that monitors a driver’s eye movements using computer vision and alerts them when signs of drowsiness are detected.

## 📌 Overview

Driver fatigue is one of the major causes of road accidents worldwide. This project uses Computer Vision and Deep Learning to detect eye closure and alert the driver before a potential accident occurs.

The system continuously monitors the driver's face through a webcam and triggers an alarm if the eyes remain closed beyond a defined threshold.

## 🎯 Features

* **👁 Real-time eye detection:** Utilizes webcam feed for continuous monitoring.
* **🧠 Deep learning-based prediction:** Accurate drowsiness detection using CNNs.
* **🔔 Instant audio alert:** Triggers a sound when drowsiness is detected.
* **⚡ Fast and lightweight:** Optimized for real-time inference.
* **🖥 Easy local setup:** Simple to run on standard hardware.

## 🛠️ Tech Stack

* **Python**
* **OpenCV**
* **NumPy**
* **Keras / TensorFlow**
* **Dlib / Haar Cascades**
* **Pygame** (for alarm sound)

## 🏗️ Project Structure

```text
DriverDrowsinessDetection/
│
├── models/                # Trained model files
├── haarcascade_files/     # Haar cascade classifiers
├── alarm.wav              # Alert sound
├── main.py                # Main execution file
├── requirements.txt       # Dependencies
└── README.md
```

## ⚙️ How It Works

### 🟢 Step 1: Capture Video
* Capture live video using the webcam.

### 🟢 Step 2: Face Detection
* Detect face using **Haar Cascade classifier**.

### 🟢 Step 3: Eye Extraction
* Extract the eye region from the detected face.

### 🟢 Step 4: Model Prediction
* Pass the eye image into the trained **CNN model**.

### 🟢 Step 5: Alert Mechanism

If eyes remain closed for consecutive frames:

* 🔔 **Trigger alarm sound**
* ⚠ **Display warning message on screen**

---

## 🚀 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/DriverDrowsinessDetection.git
cd DriverDrowsinessDetection
```
Display a warning on the screen.
2️⃣ Create Virtual Environment (Optional but Recommended)
🪟 Windows
```
python -m venv venv
venv\Scripts\activate
```
🍎 macOS / 🐧 Linux
```
python3 -m venv venv
source venv/bin/activate
```
3️⃣ Install Dependencies
```
pip install -r requirements.txt
```
4️⃣ Run the Project
```
python main.py
```
# 📊 Model Details

Architecture: Convolutional Neural Network (CNN)

Classification Type: Binary (Open Eyes / Closed Eyes)

Training Dataset: Eye image dataset

Performance: Optimized for real-time detection

# 📷 Demo Output

🟩 Green Rectangle → Eyes Open

🟥 Red Rectangle + Alarm → Drowsiness Detected

# 🔮 Future Enhancements

📱 Mobile application deployment

🚘 Vehicle system integration

🌙 Yawning detection integration

☁ Raspberry Pi / Edge deployment

📊 Driver analytics dashboard

# 📈 Applications

Smart Vehicles

Fleet Management Systems

Public Transport Monitoring

Driver Safety Assistance Systems

# 🧠 Learning Outcomes

Real-time Computer Vision implementation

CNN Model Training & Deployment

OpenCV stream processing

AI-based human safety applications

# 🤝 Contributing

Contributions are welcome!

Fork the repository

Create a new branch
```

git checkout -b feature-branch
```
Commit changes
```
git commit -m "Add new feature"
```
Push to GitHub
```
git push origin feature-branch
```
Submit a Pull Request

# 📜 License

This project is licensed under the MIT License.

# 👨‍💻 Author

## Subhash Vadaparthi
B.Tech Computer Science Engineering
Koneru Lakshmaiah Education Foundation
AI & Cloud Enthusiast
