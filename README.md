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
⚙️ How It Works
Capture live video using the webcam.

Detect face using a Haar Cascade classifier.

Extract the eye region from the detected face.

Pass the eye image to the trained CNN model.

Evaluate state: If eyes are closed for consecutive frames:

Trigger alarm sound.

Display a warning on the screen.

🚀 Installation & Setup
1️⃣ Clone the Repository

Bash
git clone [https://github.com/your-username/DriverDrowsinessDetection.git](https://github.com/your-username/DriverDrowsinessDetection.git)
cd DriverDrowsinessDetection
2️⃣ Create Virtual Environment (Optional but Recommended)

Bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
3️⃣ Install Dependencies

Bash
pip install -r requirements.txt
4️⃣ Run the Project

Bash
python main.py
📊 Model Details
Architecture: Convolutional Neural Network (CNN)

Task: Binary Classification (Open Eyes / Closed Eyes)

Training: Trained on a dedicated eye image dataset.

Performance: Optimized for high-speed, real-time inference.

📷 Demo Output
🟩 Green rectangle: Eyes Open

🟥 Red rectangle + Alarm: Drowsiness Detected

🔮 Future Improvements
📱 Deploy as a mobile application.

🚘 Integrate directly with vehicle systems.

🌙 Add yawning detection for multi-modal fatigue tracking.

☁ Deploy on edge devices like Raspberry Pi.

📊 Add a dashboard for driver analytics.

📈 Applications
Smart Vehicles

Fleet Management Systems

Public Transport Monitoring

Driver Safety Systems

🧠 Learning Outcomes
Real-time Computer Vision implementation

CNN Model Training and Deployment

OpenCV stream processing

Developing AI applications for human safety

🤝 Contributing
Contributions are welcome!

Fork the repository

Create a new branch (git checkout -b feature-branch)

Commit your changes (git commit -m "Add new feature")

Push to the branch (git push origin feature-branch)

Submit a Pull Request

📜 License
This project is open-source and available under the MIT License.

👨‍💻 Author
Subhash Vadaparthi
B.Tech Computer Science Engineering Student at Koneru Lakshmaiah Education Foundation | AI & Cloud Enthusiast
