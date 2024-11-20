# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 19:55:38 2024

@author: Shibo and Terence Laptop
"""

import streamlit as st
import joblib
import numpy as np
import os
import librosa
import cv2
import tensorflow as tf
from PIL import Image
from io import BytesIO

# Ensure the tmp directory exists
if not os.path.exists('tmp'):
    os.makedirs('tmp')

# Load models
audio_model = joblib.load(r'C:\Users\Shibo\OneDrive\Documents\AiCap\models/random_forest_bestmodel6.pkl')
image_model = joblib.load(r'C:\Users\Shibo\OneDrive\Documents\AiCap\models/image_model.pkl')

# Function to extract MFCC features from audio
def extract_mfcc(audio_bytes):
    audio, sr = librosa.load(BytesIO(audio_bytes), sr=16000)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs = np.mean(mfccs.T, axis=0)
    return mfccs.reshape(1, -1)

# Preprocess image
def preprocess_image(image_buffer):
    uploaded_image = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
    image_resized = cv2.resize(uploaded_image, (100, 100))
    image_resized_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_resized_rgb_normalized = image_resized_rgb / 255.0
    return np.expand_dims(image_resized_rgb_normalized, axis=0)

# Function to handle audio file upload
def handle_audio(file):
    st.audio(file)
    audio_bytes = file.read()
    features = extract_mfcc(audio_bytes)
    prediction = audio_model.predict(features)[0]
    probability = audio_model.predict_proba(features)[0][1]
    prediction_text = "You are HAVING A STROKE. Seek immediate medical attention!" if prediction == 1 else "You are NOT having a stroke."
    if prediction_text == "You are HAVING A STROKE. Seek immediate medical attention!":
        st.markdown(f"<div style='color:red;'>Prediction from audio: {prediction_text}</div>", unsafe_allow_html=True)
    else:
        st.write(f"Prediction from audio: {prediction_text}")
    #st.write(f"Probability of dysarthria: {probability * 100:.2f}%")
    return prediction, probability

# Function to handle image file upload
def handle_image(file):
    image_bytes = file.read()
    image_buffer = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
    st.image(file, caption='Uploaded image', width=350)
    #st.image(image, caption='Uploaded Image', use_column_width=True, width=50)  # Adjust width to smaller size
    preprocessed_image = preprocess_image(image_buffer)
    prediction = image_model.predict(preprocessed_image)
    predicted_class_index = tf.argmax(prediction, axis=1).numpy()[0]
    probability = prediction[0, predicted_class_index]
    if predicted_class_index == 0:
        probability = 1 - probability
    prediction_text = "You are HAVING A STROKE. Seek immediate medical attention!" if predicted_class_index == 1 else "You are NOT having a stroke."
    if prediction_text == "You are HAVING A STROKE. Seek immediate medical attention!":
        st.markdown(f"<div style='color:red;'>Prediction from image: {prediction_text}</div>", unsafe_allow_html=True)
    else:
        st.write(f"Prediction from image: {prediction_text}")
    #st.write(f"Probability of stroke: {probability * 100:.2f}%")
    return predicted_class_index, probability

# Streamlit app
st.set_page_config(page_title="Stroke Detection", layout="centered", page_icon="🧠")

st.markdown(
    """
    <style>
    .main {
        max-width: 800px;
        margin: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Stroke Detection")
st.write("Upload an audio file (wav) for dysarthria detection or an image file (jpg) for stroke detection.")

uploaded_file = st.file_uploader("Choose a file", type=["wav", "jpg", "jpeg"])

if uploaded_file:
    file_type = uploaded_file.type
    prediction1 = None
    probability1 = None

    if "audio" in file_type:
        st.header("Audio File Upload")
        prediction1, probability1 = handle_audio(uploaded_file)
    elif "image" in file_type:
        st.header("Image File Upload")
        prediction1, probability1 = handle_image(uploaded_file)

    uploaded_file2 = st.file_uploader("Choose another file (optional, should be of different type)...", type=["wav", "jpg", "jpeg"], key="file2")

    if uploaded_file2:
        file_type2 = uploaded_file2.type
        if ("audio" in file_type and "image" in file_type2) or ("image" in file_type and "audio" in file_type2):
            if "audio" in file_type2:
                st.header("Second File: Audio File Upload")
                prediction2, probability2 = handle_audio(uploaded_file2)
            elif "image" in file_type2:
                st.header("Second File: Image File Upload")
                prediction2, probability2 = handle_image(uploaded_file2)

            if prediction1 is not None and prediction2 is not None:
                if prediction1 == 1 and prediction2 == 1:
                    st.markdown("<div style='color:red; font-size:20px;'>Final prediction: Both predictions indicate a stroke.</div>", unsafe_allow_html=True)
                elif prediction1 == 0 and prediction2 == 0:
                    st.write("Final prediction: Both predictions indicate no stroke.")
                else:
                    st.markdown("<div style='color:red; font-size:20px;'>Final prediction: You are HAVING A STROKE. Seek immediate medical attention!</div>", unsafe_allow_html=True)
        else:
            st.write("Please upload a different file type for a combined prediction (image and audio).")
