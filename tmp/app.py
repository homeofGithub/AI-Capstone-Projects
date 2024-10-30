from flask import Flask, request, render_template
import joblib
import numpy as np
import os
import librosa

app = Flask(__name__)

# Ensure the tmp directory exists
if not os.path.exists('tmp'):
    os.makedirs('tmp')

# Load the saved model
model = joblib.load('models/random_forest_bestmodel.pkl')

# Function to extract MFCC features from audio
def extract_mfcc(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs = np.mean(mfccs.T, axis=0)
    return mfccs.reshape(1, -1)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file and file.filename.endswith('.wav'):
        # Save the file to a temporary location
        filepath = os.path.join('tmp', file.filename)
        file.save(filepath)
        features = extract_mfcc(filepath)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        result = {
            'prediction': int(prediction),
            'probability': round(probability * 100, 2)
        }
        return render_template('index.html', result=result)
    return "Invalid file format"

if __name__ == '__main__':
    app.run(debug=True)
