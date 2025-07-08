# 🛡️ Raspberry Pi-Based Self-Defense System with AI

A real-time self-defense and alert system built using **Raspberry Pi**, combining motion and sound detection, GPS-based location alerts, image/audio capture, and advanced **machine learning models** for facial expression and audio threat recognition.

---

## 📌 Features

- 🎯 **Real-Time Threat Detection** using motion (PIR) and ultrasonic sensors.
- 🧠 **Facial Expression Recognition** using a CNN model trained on FER2013.
- 🔊 **Audio Threat Detection** using CNN + LSTM trained on scream, cry, gunshot sounds.
- 📍 **Location Awareness** with GPS for precise emergency SMS alerts.
- 📷 **Camera Capture** with AWS S3 image storage.
- 📡 **SMS Notification** using GSM module for instant emergency alerts.
- 💾 Offline-capable with local processing on Raspberry Pi.

---

## 🧱 System Architecture

Sensors (Motion + Ultrasonic + Mic)
↓
Raspberry Pi
(Python + ML)
↓
Pi Camera | GPS | GSM
↓ ↓ ↓
Image Location SMS Alert
↓
Upload to AWS S3

yaml
Copy
Edit

---

## 📂 Project Structure

self_defense_system/
├── self_defense_system.py # Main threat detection & response script
├── train_face_expression.py # CNN training for facial emotion
├── train_sound_classifier.py # CNN+LSTM model training for sound
├── models/
│ ├── expr_cnn.h5 # Trained facial expression model
│ └── audio_cnn_lstm.h5 # Trained sound classification model
├── audio_data/ # Training data for sound recognition
├── fer2013.csv # Dataset for expression training
└── README.md # Project documentation

yaml
Copy
Edit

---

## 🛠️ Hardware Requirements

- ✅ Raspberry Pi 3/4 (with Raspbian OS)
- 📷 Pi Camera
- 🎤 Microphone (USB or Pi-compatible)
- 🛰️ GPS module (e.g., NEO-6M)
- 📡 GSM module (e.g., SIM800L)
- 🔌 PIR motion sensor + Ultrasonic sensor
- ⚡ Power supply

---

## 🧪 Software Setup

### 🔧 Dependencies

```bash
pip install tensorflow numpy pandas boto3 opencv-python librosa
sudo apt install python3-picamera
🚀 Getting Started
🔍 1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-username/self-defense-system.git
cd self-defense-system
📁 2. Train the Models (Optional)
Facial Expression (FER2013)
bash
Copy
Edit
python train_face_expression.py
Sound Classifier
bash
Copy
Edit
python train_sound_classifier.py
Or download pre-trained models and place in models/ folder.

🟢 Run the System
bash
Copy
Edit
sudo python3 self_defense_system.py
The system will continuously monitor for threats and respond instantly by:

Capturing an image

Predicting emotion

Sending SMS with GPS location and S3 image URL

Optionally, analyzing sound

🧠 Datasets Used
FER2013 Emotion Dataset (Kaggle)

Custom audio dataset (./audio_data/) with scream, cry, gunshot, and normal sounds.

📦 Output
✅ Emotion label + confidence

✅ Sound label (optional)

✅ GPS Coordinates

✅ 📷 Image uploaded to AWS S3

✅ 📩 SMS to predefined contacts
