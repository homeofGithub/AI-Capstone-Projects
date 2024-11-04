#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 16:50:37 2024

@author: bettyhan
"""

from flask import Flask, request, render_template
import joblib
import numpy as np
import os
import librosa
import tensorflow as tf
import moviepy.editor as mp
import cv2
import os

app = Flask(__name__)

# Ensure the tmp directory exists
if not os.path.exists('tmp'):
    os.makedirs('tmp')

# Load the saved model
model = joblib.load('models/random_forest_bestmodel6.pkl')

# Load the saved model
model_image = joblib.load('models/image_model.pkl')

#Function to extract audio and image from video
def extract_audio(video_file, audio_output):
    # Load the video file
    video = mp.VideoFileClip(video_file)
    # Write the audio to a WAV file
    video.audio.write_audiofile(audio_output, codec='pcm_s16le')
    print(f"Audio extracted to {audio_output}")

def extract_images(video_file, image_folder, interval=1):
    # Create a folder to save images if it doesn't exist
    os.makedirs(image_folder, exist_ok=True)

    # Open the video file with OpenCV
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)

    count = 0
    success = True
    while success:
        # Read a frame from the video
        success, frame = cap.read()
        if success:
            # Save the frame as an image
            image_path = os.path.join(image_folder, f"frame_{count:04d}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"Image saved to {image_path}")
            count += 1
            
            # Skip frames based on the interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, count * int(fps * interval))

    cap.release()
    print(f"Images extracted to {image_folder}")
# Function to extract MFCC features from audio
def extract_mfcc(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs = np.mean(mfccs.T, axis=0)
    return mfccs.reshape(1, -1)

# Function to Preprocess image
def preprocess_image(image_path):
    image = cv2.imread(image_path) # Read in image
    image_resized = cv2.resize(image, (100 , 100)) # Resize
    image_resized_rgb =  cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB) # Convert colour from BGR (cv2's default) to RGB
    image_resized_rgb_normalized = image_resized_rgb/255.0 # Normalization
    
    # Current shape: (100, 100, 3)
    # CNN model expects arrays of (batch_size, height, width, channels), even for single image inputs
    # Use expand_dims() to add a new dimension at the beginning (axis=0) of the array 
    preprocessed_image = np.expand_dims(image_resized_rgb_normalized, axis=0)
    
    # Current shape: (1, 100, 100, 3)

    return preprocessed_image


@app.route('/', methods=['GET'])
def index():
    return render_template('index_video.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file and file.filename.endswith('.mp4'):
        # Save the file to a temporary location
        filepath = os.path.join('tmp', file.filename)
        file.save(filepath)

        audio_output = os.path.join('tmp', 'audio.wav')
        image_folder = 'tmp'

        extract_audio(filepath, audio_output)
        extract_images(filepath, image_folder, interval=30)  # Extracts one frame per second
 
        ############# audio prediction ##############
        features = extract_mfcc(audio_output)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]   
 
        ############ image prediction##############
        #Preprocess the image
        image_output= os.path.join('tmp', 'frame_0000.jpg')
        print(image_output)
        image = preprocess_image(image_output)
        print(image.shape)
        # Make prediction
        prediction_image = model_image.predict(image)

        # Predicted class is the element with the highest value
        predicted_class_index = tf.argmax(prediction_image, axis=1)
        predicted_class_index = predicted_class_index.numpy()[0]
        
        # Sample prediction result: [0.9961105  0.00388956]
        # Index with the highest probability is row 0, column 0
        probability_image = prediction_image[0, predicted_class_index] # row number 0, column number predict_class_index

        print(predicted_class_index)
        print(probability_image)

        if (predicted_class_index == 0):
            probability_image = 1 - probability_image        
        
 
        result = {
            'prediction': int(prediction),
            'probability': round(probability * 100, 2),
            'prediction_image': predicted_class_index,
            'probability_image': round(probability_image * 100, 2)
        }
        return render_template('index_video.html', result=result)
    return "Invalid file format"

if __name__ == '__main__':
    app.run(debug=True)
