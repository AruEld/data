import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
import io

# Load model

cnn_model = tf.keras.models.load_model("cnn_motor_fault.keras")

# File uploader
uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])
if uploaded_file:
    y, sr = librosa.load(uploaded_file, sr=16000)
    
    # Generate and display spectrogram
    st.audio(uploaded_file)
    st.write("### Mel Spectrogram")
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_dB, sr=sr, ax=ax)
    st.pyplot(fig)

    # Prepare input
    S_resized = tf.image.resize(S_dB[..., np.newaxis], (128, 256)).numpy()
    S_input = np.expand_dims(S_resized, axis=0)

    # Predict
    preds = cnn_model.predict(S_input)
    pred_class = np.argmax(preds)

    st.write(f"### Predicted Class: {pred_class}")
