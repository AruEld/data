import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import io
import tempfile

st.title("🔊 Motor Fault Classifier")

# Upload the model manually
uploaded_model = st.file_uploader("Upload CNN model (.h5)", type=["h5"])
if uploaded_model:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
        tmp.write(uploaded_model.read())
        model_path = tmp.name
    cnn_model = tf.keras.models.load_model(model_path)

    # Upload the audio
    uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])
    if uploaded_file:
        st.audio(uploaded_file)

        y, sr = librosa.load(uploaded_file, sr=16000)
        S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)

        st.write("### Mel Spectrogram")
        fig, ax = plt.subplots()
        img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        st.pyplot(fig)

        S_resized = tf.image.resize(S_dB[..., np.newaxis], (128, 256)).numpy()
        S_input = np.expand_dims(S_resized, axis=0)

        preds = cnn_model.predict(S_input)
        pred_class_idx = int(np.argmax(preds))
        confidence = float(np.max(preds))
        label_names = ["off", "on", "cap", "out", "unb", "c75", "vnt"]
        pred_class_label = label_names[pred_class_idx] if pred_class_idx < len(label_names) else f"Class {pred_class_idx}"

        st.write(f"### 🧠 Predicted Class: `{pred_class_label}`")
        st.write(f"Confidence: `{confidence:.2f}`")
