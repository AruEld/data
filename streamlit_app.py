import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import requests
import tempfile
https://drive.google.com/file/d/1JmT-NseWZr0F9GBZlmrqMMerSNsMW4TX/view?usp=sharing
def load_model_from_gdrive(file_id):
    url = f"https://drive.google.com/file/d/1JmT-NseWZr0F9GBZlmrqMMerSNsMW4TX'
    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
        response = requests.get(url)
        tmp.write(response.content)
        tmp.flush()
        model = tf.keras.models.load_model(tmp.name, compile=False)
    return model


# 🔑 Replace with your actual Google Drive file ID
FILE_ID = "YOUR_FILE_ID_HERE"  # 🔁 Paste the correct file ID
cnn_model = load_model_from_gdrive(FILE_ID)

# 🏷️ Class labels (update to match training)
label_names = ["off", "on", "cap", "out", "unb", "c25", "c75", "vnt"]

# 🎯 Streamlit App
st.title("🔊 Motor Fault Classifier")
st.write("Upload a WAV file to classify the motor fault condition using a CNN model trained on MFCC features.")

# 📤 Upload .wav file
uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file)

    # 🎧 Extract MFCC
    y, sr = librosa.load(uploaded_file, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    # 🧹 Padding or truncating
    max_len = 173
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]

    # 📊 Show MFCC
    st.write("### MFCC (Mel-Frequency Cepstral Coefficients)")
    fig, ax = plt.subplots()
    img = librosa.display.specshow(mfcc, sr=sr, x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax)
    st.pyplot(fig)

    # 🔍 Predict
    mfcc_input = np.expand_dims(mfcc[..., np.newaxis], axis=0)  # Shape: (1, 40, 173, 1)
    preds = cnn_model.predict(mfcc_input)
    pred_idx = int(np.argmax(preds))
    confidence = float(np.max(preds))

    # 📢 Output
    label = label_names[pred_idx] if pred_idx < len(label_names) else f"Class {pred_idx}"
    st.write(f"### ✅ Predicted Class: `{label}`")
    st.write(f"Confidence: `{confidence:.2f}`")
