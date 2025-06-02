import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf

# 🧠 Load trained model (make sure cnn_motor_fault.keras is in the same directory)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cnn_motors_fault.keras")

model = load_model()

# 🏷️ Class label names (must match training order)
label_names = ["off", "on", "cap", "out", "unb", "c25", "c75", "vnt"]

st.title("🔊 Motor Fault Classifier")
st.write("Upload a WAV file to classify motor fault condition using a CNN model trained on MFCC features.")

# 📤 File upload
uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])
if uploaded_file:
    st.audio(uploaded_file)

    # 🎧 Load and preprocess audio
    y, sr = librosa.load(uploaded_file, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    max_len = 173
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]

    # 🎛️ Display MFCC spectrogram
    st.write("### Mel-frequency Cepstral Coefficients (MFCC)")
    fig, ax = plt.subplots()
    img = librosa.display.specshow(mfcc, sr=sr, x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax)
    st.pyplot(fig)

    # 🧠 Predict
    mfcc_input = np.expand_dims(mfcc[..., np.newaxis], axis=0)  # (1, 40, 173, 1)
    preds = model.predict(mfcc_input)
    pred_idx = np.argmax(preds)
    confidence = float(np.max(preds))
    label = label_names[pred_idx] if pred_idx < len(label_names) else f"Class {pred_idx}"

    # 📝 Display result
    st.write(f"### ✅ Predicted Class: `{label}`")
    st.write(f"Confidence: `{confidence:.2f}`")
