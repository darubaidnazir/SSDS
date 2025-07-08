import RPi.GPIO as GPIO
import serial
import time
import numpy as np
import boto3
import cv2
from picamera import PiCamera
from tensorflow.keras.models import load_model
from datetime import datetime

# === CONFIGURATION ===
MOTION_PIN = 17
TRIGGER_PIN = 23
ECHO_PIN = 24
GSM_PORT = "/dev/ttyS0"
CONTACTS = ['+91XXXXXXXXXX']  # Replace with your phone numbers
AWS_BUCKET = 'your-s3-bucket-name'  # Replace with your S3 bucket
IMAGE_PATH = '/tmp/self_defense_img.jpg'
EXPR_MODEL_PATH = 'models/expr_cnn.h5'  # Replace with path to your model

# === INITIALIZE ===
GPIO.setmode(GPIO.BCM)
GPIO.setup(MOTION_PIN, GPIO.IN)
GPIO.setup(TRIGGER_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)
camera = PiCamera()
s3 = boto3.client('s3')
expr_model = load_model(EXPR_MODEL_PATH)

# === FUNCTIONS ===
def get_distance():
    GPIO.output(TRIGGER_PIN, True)
    time.sleep(0.00001)
    GPIO.output(TRIGGER_PIN, False)

    while GPIO.input(ECHO_PIN) == 0:
        pulse_start = time.time()

    while GPIO.input(ECHO_PIN) == 1:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start
    return pulse_duration * 17150

def capture_image():
    camera.capture(IMAGE_PATH)
    return IMAGE_PATH

def upload_to_s3(file_path):
    s3_key = f"events/{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    s3.upload_file(file_path, AWS_BUCKET, s3_key)
    return f"https://{AWS_BUCKET}.s3.amazonaws.com/{s3_key}"

def predict_expression(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))
    img = img.reshape(1, 48, 48, 1) / 255.0
    preds = expr_model.predict(img)
    labels = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    return labels[np.argmax(preds)], float(np.max(preds))

def send_sms(serial_port, contacts, message):
    ser = serial.Serial(serial_port, baudrate=9600, timeout=1)
    ser.write(b'AT+CMGF=1\r')
    time.sleep(0.5)
    for contact in contacts:
        ser.write(f'AT+CMGS="{contact}"\r'.encode())
        time.sleep(0.5)
        ser.write(message.encode() + b"\x1A")
        time.sleep(3)

def on_threat_detected():
    print("🔴 Threat detected!")
    image_path = capture_image()
    img_url = upload_to_s3(image_path)
    emotion, conf = predict_expression(image_path)
    msg = f"⚠️ ALERT!\nEmotion: {emotion} ({conf:.2f})\nImage: {img_url}"
    print(msg)
    send_sms(GSM_PORT, CONTACTS, msg)

# === MAIN LOOP ===
try:
    print("🟢 Self-Defense System Armed")
    while True:
        if GPIO.input(MOTION_PIN) or get_distance() < 50:
            on_threat_detected()
            time.sleep(10)  # Avoid spamming alerts
except KeyboardInterrupt:
    print("🛑 System Deactivated")
finally:
    GPIO.cleanup()
