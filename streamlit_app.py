import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import io
import requests
import tempfile

st.title("Motor Fault Classifier")

# 🔁 Google Drive direct download helper
def gdrive_to_direct_link(gdrive_url):
    file_id = gdrive_url.split("/d/")[1].split("/")[0]
    return f"https://drive.google.com/uc?export=download&id={file_id}"

# ✅ Model loading
gdrive_url = "https://drive.google.com/file/d/1BbtgHJ08pjOwMPex2WBKgtxwDzZ89N1V/view?usp=sharing"
download_url = gdrive_to_direct_link(gdrive_url)

with tempfile.NamedTemporaryFile(suffix=".keras") as tmp:
    response = requests.get(download_url)
    tmp.write(response.content)
    tmp.flush()
    cnn_model = tf.keras.models.load_model(tmp.name)

# 📤 File uploader
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
if uploaded_file:
    st.audio(uploaded_file)

    # 🎧 Load audio
    y, sr = librosa.load(uploaded_file, sr=16000)
    
    # 🎛️ Generate mel spectrogram
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # 📊 Show spectrogram
    st.write("### Mel Spectrogram")
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_dB, sr=sr, ax=ax, x_axis='time', y_axis='mel')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    st.pyplot(fig)

    # 🧠 Preprocess and predict
    S_resized = tf.image.resize(S_dB[..., np.newaxis], (128, 256)).numpy()
    S_input = np.expand_dims(S_resized, axis=0)
    preds = cnn_model.predict(S_input)
    pred_class = int(np.argmax(preds))
    confidence = float(np.max(preds))

    st.write(f"### Predicted Class: `{pred_class}` (Confidence: {confidence:.2f})")
