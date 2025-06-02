import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import tempfile

st.set_page_config(page_title="Motor Fault Classifier")
st.title("🔊 Motor Fault Classifier (weights-based)")

LABELS = ["off", "on", "cap", "out", "unb", "c75", "vnt"]

# ✅ Rebuild model architecture (match training)
def build_model():
    input_layer = tf.keras.Input(shape=(40, 173, 1))
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(len(LABELS), activation='softmax')(x)
    model = tf.keras.Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 📤 Upload CNN weights
uploaded_model = st.file_uploader("Upload CNN Weights (.h5)", type=["h5"])
cnn_model = None
if uploaded_model:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
        tmp.write(uploaded_model.read())
        model_path = tmp.name

    try:
        cnn_model = build_model()
        cnn_model.load_weights(model_path)
        st.success("✅ Weights loaded and model built successfully.")
    except Exception as e:
        st.error(f"❌ Failed to load weights: {e}")

# 📤 Upload audio for prediction
if cnn_model:
    uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
    if uploaded_file:
        st.audio(uploaded_file)

        # Load and preprocess
        y, sr = librosa.load(uploaded_file, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        if mfcc.shape[1] < 173:
            pad = 173 - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad)), mode='constant')
        else:
            mfcc = mfcc[:, :173]
        mfcc = mfcc[..., np.newaxis]
        input_tensor = np.expand_dims(mfcc, axis=0)

        # Visualize
        st.write("### 🎛 Mel Spectrogram")
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        fig, ax = plt.subplots()
        img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        st.pyplot(fig)

        # Predict
        preds = cnn_model.predict(input_tensor)
        pred_class = int(np.argmax(preds))
        confidence = float(np.max(preds))
        label = LABELS[pred_class] if pred_class < len(LABELS) else f"Class {pred_class}"

        st.markdown(f"### 🧠 Predicted Class: `{label}`")
        st.markdown(f"**Confidence:** `{confidence:.2f}`")
