🚗 Driver Drowsiness Detection System

An AI-powered real-time Driver Drowsiness Detection System that monitors a driver’s eye movements using computer vision and alerts them when signs of drowsiness are detected.

📌 Overview

Driver fatigue is one of the major causes of road accidents worldwide.
This project uses Computer Vision and Deep Learning to detect eye closure and alert the driver before a potential accident occurs.

The system continuously monitors the driver's face through a webcam and triggers an alarm if the eyes remain closed beyond a defined threshold.

🎯 Features

👁 Real-time eye detection using webcam

🧠 Deep learning-based drowsiness prediction

🔔 Instant audio alert when drowsiness detected

⚡ Fast and lightweight implementation

🖥 Easy to run locally

🛠️ Tech Stack

Python

OpenCV

NumPy

Keras / TensorFlow

Dlib / Haar Cascades

Pygame (for alarm sound)

🏗️ Project Structure
DriverDrowsinessDetection/
│
├── models/                # Trained model files
├── haarcascade_files/     # Haar cascade classifiers
├── alarm.wav              # Alert sound
├── main.py                # Main execution file
├── requirements.txt       # Dependencies
└── README.md
⚙️ How It Works

Capture live video using webcam.

Detect face using Haar Cascade classifier.

Extract eye region from detected face.

Pass eye image to trained CNN model.

If eyes are closed for consecutive frames:

Trigger alarm sound.

Display warning on screen.

🚀 Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/your-username/DriverDrowsinessDetection.git
cd DriverDrowsinessDetection
2️⃣ Create Virtual Environment (Optional but Recommended)
python -m venv venv
venv\Scripts\activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Run the Project
python main.py
📊 Model Details

Convolutional Neural Network (CNN)

Binary Classification:

Open Eyes

Closed Eyes

Trained on eye image dataset

Optimized for real-time inference

📷 Demo Output

Green rectangle → Eyes Open

Red rectangle + Alarm → Drowsiness Detected

🔮 Future Improvements

📱 Deploy as mobile application

🚘 Integrate with vehicle system

🌙 Add yawning detection

☁ Deploy on edge devices (Raspberry Pi)

📊 Add dashboard analytics

📈 Applications

Smart Vehicles

Fleet Management Systems

Public Transport Monitoring

Driver Safety Systems

🧠 Learning Outcomes

Real-time Computer Vision

CNN Model Training

OpenCV Implementation

Human Safety AI Applications

🤝 Contributing

Contributions are welcome!

Fork the repository

Create a new branch

Commit your changes

Submit a Pull Request

📜 License

This project is open-source and available under the MIT License.

👨‍💻 Author

Subhash Vadaparthi
B.Tech Computer Science Engineering
AI & Cloud Enthusiast
