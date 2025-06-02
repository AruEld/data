import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import io
import tempfile
import gdown
import os

st.title("🔊 Motor Fault Classifier")

# 🔁 Step 1: Download model from Google Drive using gdown
@st.cache_resource
def load_model_from_gdrive():
    file_id = "1BbtgHJ08pjOwMPex2WBKgtxwDzZ89N1V"
    url = f"https://drive.google.com/uc?id={file_id}"

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "cnn_motor_fault.keras")
        gdown.download(url, model_path, quiet=False)
        model = tf.keras.models.load_model(model_path)
    return model

cnn_model = load_model_from_gdrive()

# 🧾 Label index mapping (adjust if needed)
label_names = ["off", "on", "cap", "out", "unb", "c75", "vnt"]

# 📤 File uploader
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
if uploaded_file:
    st.audio(uploaded_file)

    # 🎧 Load audio
    y, sr = librosa.load(uploaded_file, sr=16000)

    # 🎛️ Mel spectrogram
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    st.write("### Mel Spectrogram")
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    st.pyplot(fig)

    # 🧠 Preprocess and predict
    S_resized = tf.image.resize(S_dB[..., np.newaxis], (128, 256)).numpy()
    S_input = np.expand_dims(S_resized, axis=0)

    preds = cnn_model.predict(S_input)
    pred_class_idx = int(np.argmax(preds))
    confidence = float(np.max(preds))
    pred_class_label = label_names[pred_class_idx] if pred_class_idx < len(label_names) else f"Class {pred_class_idx}"

    st.write(f"### 🧠 Predicted Class: `{pred_class_label}`")
    st.write(f"Confidence: `{confidence:.2f}`")
