import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout

def extract_features(file_path, max_len=22050):
    y, sr = librosa.load(file_path, sr=22050)
    if len(y) > max_len:
        y = y[:max_len]
    else:
        y = np.pad(y, (0, max_len - len(y)))
    mfcc = librosa.feature.mfcc(y, sr=sr, n_mfcc=40)
    return mfcc.T  # Shape: (time, features)

X, y = [], []
class_map = {'scream': 0, 'cry': 1, 'gunshot': 2, 'normal': 3}

# Load your dataset: ./audio_data/{label}/file.wav
for label in class_map:
    folder = f'./audio_data/{label}/'
    for f in os.listdir(folder):
        features = extract_features(os.path.join(folder, f))
        X.append(features)
        y.append(class_map[label])

X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=44, dtype='float32', padding='post')
y = tf.keras.utils.to_categorical(y, num_classes=4)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(44, 40)),
    MaxPooling1D(2),
    Dropout(0.3),
    LSTM(64),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))
model.save('models/audio_cnn_lstm.h5')
