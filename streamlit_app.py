import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import tempfile

st.set_page_config(page_title="Motor Fault Classifier")
st.title("🔊 Motor Fault Classifier")

# 📤 Upload H5 model
uploaded_model = st.file_uploader("Upload CNN model (.h5)", type=["h5"])
if uploaded_model:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
        tmp.write(uploaded_model.read())
        tmp_path = tmp.name

    # ⛔ Bypass legacy incompatibility using compile=False
    try:
        cnn_model = tf.keras.models.load_model(tmp_path, compile=False)
        st.success("✅ Model loaded successfully.")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

    # 📁 Upload .wav file
    uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
    if uploaded_file:
        st.audio(uploaded_file)

        y, sr = librosa.load(uploaded_file, sr=16000)
        S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)

        st.write("### 🎛 Mel Spectrogram")
        fig, ax = plt.subplots()
        img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        st.pyplot(fig)

        # 🧠 Prepare and Predict
        S_resized = tf.image.resize(S_dB[..., np.newaxis], (128, 256)).numpy()
        S_input = np.expand_dims(S_resized, axis=0)

        preds = cnn_model.predict(S_input)
        pred_idx = int(np.argmax(preds))
        confidence = float(np.max(preds))
        labels = ["off", "on", "cap", "out", "unb", "c75", "vnt"]
        label = labels[pred_idx] if pred_idx < len(labels) else f"Class {pred_idx}"

        st.markdown(f"### 🧠 Predicted Class: `{label}`")
        st.markdown(f"**Confidence:** `{confidence:.2f}`")
