# Sound inference
from tensorflow.keras.models import load_model
import librosa

sound_model = load_model('models/audio_cnn_lstm.h5')

def predict_sound_class(audio_path):
    features = extract_features(audio_path)
    features = tf.keras.preprocessing.sequence.pad_sequences([features], maxlen=44, dtype='float32', padding='post')
    pred = sound_model.predict(features)
    labels = ['scream', 'cry', 'gunshot', 'normal']
    return labels[np.argmax(pred)], float(np.max(pred))
