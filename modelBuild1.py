# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 13:41:16 2024

@author: Shibo
"""

import os
import librosa
import numpy as np
from pydub import AudioSegment
import wave

def load_audio_files1(directory):
    audio_files = []
    labels = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                audio, sr = librosa.load(file_path, sr=16000)
                audio_files.append(audio)
                # Assign labels based on directory structure
                
                labels.append(1)  # Dysarthria
                
    return audio_files, labels
def load_audio_files1(directory):
    audio_files = []
    labels = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                
                try:
                    # Check if the file is a valid wav file using wave module
                    with wave.open(file_path, 'rb') as wf:
                        wf.readframes(1)
                    
                    # Attempt to load using librosa
                    audio, sr = librosa.load(file_path, sr=16000)
                except wave.Error as we:
                    print(f"Wave module failed for {file_path}: {we}")
                except Exception as e:
                    print(f"Librosa load failed for {file_path}: {e}")
                    try:
                        # Attempt to convert using ffmpeg
                        os.system(f'ffmpeg -i "{file_path}" -ar 16000 -ac 1 "{file_path}.converted.wav"')
                        # Load the converted file
                        audio, sr = librosa.load(f'{file_path}.converted.wav', sr=16000)
                    except Exception as e:
                        print(f"Conversion with ffmpeg failed for {file_path}: {e}")
                        continue
                
                audio_files.append(audio)
                # Assign labels based on directory structure
                labels.append(1)  # Dysarthria

    return audio_files, labels


def load_audio_files2(directory):
    audio_files = []
    labels = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                audio, sr = librosa.load(file_path, sr=16000)
                audio_files.append(audio)
                # Assign labels based on directory structure
                
                labels.append(0)  # None Dysarthria
                
    return audio_files, labels

# Load audio files and labels
audio_filesFF011, labelsFF011 = load_audio_files1(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\F\F01\Session1\wav_arrayMic')
audio_filesFF011h, labelsFF011h = load_audio_files1(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\F\F01\Session1\wav_headMic')

audio_filesFF031, labelsFF031 = load_audio_files1(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\F\F03\Session1\wav_arrayMic')
audio_filesFF031h, labelsFF031h = load_audio_files1(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\F\F03\Session1\wav_headMic')

audio_filesFF032, labelsFF032 = load_audio_files1(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\F\F03\Session2\wav_arrayMic')
audio_filesFF032h, labelsFF032h = load_audio_files1(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\F\F03\Session2\wav_headMic')
audio_filesFF032a, labelsFF032a = load_audio_files1(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\F\F03\Session2\wavall')

audio_filesFF033, labelsFF033 = load_audio_files1(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\F\F03\Session3\wav_arrayMic')
audio_filesFF033h, labelsFF033h = load_audio_files1(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\F\F03\Session3\wav_headMic')
audio_filesFF033a, labelsFF033a = load_audio_files1(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\F\F03\Session3\wavall')

audio_filesFF041, labelsFF041 = load_audio_files1(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\F\F04\Session1\wav_arrayMic')
audio_filesFF042, labelsFF042 = load_audio_files1(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\F\F04\Session2\wav_arrayMic')
audio_filesFF042h, labelsFF042h = load_audio_files1(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\F\F04\Session2\wav_headMic')
audio_filesFF042a, labelsFF042a = load_audio_files1(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\F\F04\Session2\wavall')

audio_filesMM011, labelsMM011 = load_audio_files1(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\M\M01\Session1\wav_arrayMic')
audio_filesMM011h, labelsMM011h = load_audio_files1(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\M\M01\Session1\wav_headMic')
audio_filesMM011a, labelsMM011a = load_audio_files1(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\M\M01\Session1\wavall')

audio_filesMM012, labelsMM012 = load_audio_files1(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\M\M01\Session2_3\wav_arrayMic')
audio_filesMM012h, labelsMM012h = load_audio_files1(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\M\M01\Session2_3\wav_headMic')

audio_filesMM021, labelsMM021 = load_audio_files1(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\M\M02\Session1\wav_arrayMic')
audio_filesMM021h, labelsMM021h = load_audio_files1(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\M\M02\Session1\wav_headMic')

audio_filesMM022, labelsMM022 = load_audio_files1(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\M\M02\Session2\wav_arrayMic')
audio_filesMM022h, labelsMM022h = load_audio_files1(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\M\M02\Session2\wav_headMic')
audio_filesMM022a, labelsMM022a = load_audio_files1(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\M\M02\Session2\wavall')

audio_filesMM032, labelsMM032 = load_audio_files1(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\M\M03\Session2\wav_arrayMic')
audio_filesMM032h, labelsMM032h = load_audio_files1(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\M\M03\Session2\wav_headMic')
audio_filesMM032a, labelsMM032a = load_audio_files1(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\M\M03\Session2\wavall')

audio_filesMM041, labelsMM041 = load_audio_files1(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\M\M04\Session1\wav_arrayMic')
audio_filesMM042, labelsMM042 = load_audio_files1(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\M\M04\Session2\wav_arrayMic')
audio_filesMM042h, labelsMM042h = load_audio_files1(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\M\M04\Session2\wav_headMic')
audio_filesMM042a, labelsMM042a = load_audio_files1(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\M\M04\Session2\wavall')

audio_filesMM051, labelsMM051 = load_audio_files1(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\M\M05\Session1\wav_arrayMic')
audio_filesMM051h, labelsMM051h = load_audio_files1(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\M\M05\Session1\wav_headMic')
audio_filesMM051a, labelsMM051a = load_audio_files1(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\M\M05\Session1\wavall')

audio_filesMM052, labelsMM052 = load_audio_files1(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\M\M05\Session2\wav_headMic')
audio_filesMM052a, labelsMM052a = load_audio_files1(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\M\M05\Session2\wavall')

audio_filesFC011, labelsFC011 = load_audio_files2(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\FC\FC01\Session1\wav_arrayMic')
audio_filesFC022, labelsFC022 = load_audio_files2(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\FC\FC02\Session2\wav_arrayMic')
audio_filesFC023, labelsFC023 = load_audio_files2(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\FC\FC02\Session3\wav_arrayMic')
audio_filesFC031, labelsFC031 = load_audio_files2(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\FC\FC03\Session1\wav_arrayMic')
audio_filesFC032, labelsFC032 = load_audio_files2(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\FC\FC03\Session2\wav_arrayMic')
audio_filesFC033, labelsFC033 = load_audio_files2(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\FC\FC03\Session3\wav_arrayMic')
audio_filesMC011, labelsMC011 = load_audio_files2(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\MC\MC01\Session1\wav_arrayMic')
audio_filesMC012, labelsMC012 = load_audio_files2(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\MC\MC01\Session2\wav_arrayMic')
audio_filesMC013, labelsMC013 = load_audio_files2(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\MC\MC01\Session3\wav_arrayMic')
audio_filesMC021, labelsMC021 = load_audio_files2(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\MC\MC02\Session1\wav_arrayMic')
audio_filesMC022, labelsMC022 = load_audio_files2(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\MC\MC02\Session2\wav_arrayMic')
audio_filesMC031, labelsMC031 = load_audio_files2(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\MC\MC03\Session1\wav_arrayMic')
audio_filesMC032, labelsMC032 = load_audio_files2(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\MC\MC03\Session2\wav_arrayMic')
audio_filesMC041, labelsMC041 = load_audio_files2(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\MC\MC04\Session1\wav_arrayMic')
audio_filesMC042, labelsMC042 = load_audio_files2(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\MC\MC04\Session2\wav_arrayMic')

audio_filesH, labelsH = load_audio_files2(r'C:\Users\Shibo\OneDrive\Documents\AiCap\RNN\H')
audio_filesL, labelsL = load_audio_files2(r'C:\Users\Shibo\Desktop\download\wav')

# Combine audio files and labels
total_audio_files = (audio_filesFF011 + audio_filesFF031 + audio_filesFF032 + audio_filesFF033 + audio_filesFF041 + audio_filesFF042 +
                     audio_filesMM011 + audio_filesMM012 + audio_filesMM021 + audio_filesMM022 + audio_filesMM032 + audio_filesMM041 +
                     audio_filesMM042 + audio_filesMM051 + audio_filesMM052 + audio_filesFC011 + audio_filesFC022 + audio_filesFC023 +
                     audio_filesFC031 + audio_filesFC032 + audio_filesFC033 + audio_filesMC011 + audio_filesMC012 + audio_filesMC013 +
                     audio_filesMC021 + audio_filesMC022 + audio_filesMC031 + audio_filesMC032 + audio_filesMC041 + audio_filesMC042 + 
                     audio_filesH + audio_filesL + audio_filesFF011h+audio_filesFF031h+audio_filesFF032h+audio_filesFF032a+audio_filesFF033h+audio_filesFF033a+audio_filesFF042h+audio_filesFF042a+audio_filesMM011h+audio_filesMM011a+
                     audio_filesMM012h+audio_filesMM021h+audio_filesMM022h+audio_filesMM022a+audio_filesMM032h+audio_filesMM032a+audio_filesMM042h+audio_filesMM042a+audio_filesMM051h+audio_filesMM051a+audio_filesMM052a)

total_labels = (labelsFF011 + labelsFF031 + labelsFF032 + labelsFF033 + labelsFF041 + labelsFF042 +
                labelsMM011 + labelsMM012 + labelsMM021 + labelsMM022 + labelsMM032 + labelsMM041 +
                labelsMM042 + labelsMM051 + labelsMM052 + labelsFC011 + labelsFC022 + labelsFC023 +
                labelsFC031 + labelsFC032 + labelsFC033 + labelsMC011 + labelsMC012 + labelsMC013 +
                labelsMC021 + labelsMC022 + labelsMC031 + labelsMC032 + labelsMC041 + labelsMC042 + 
                labelsH + labelsL + labelsFF011h+labelsFF031h+labelsFF032h+labelsFF032a+labelsFF033h+labelsFF033a+labelsFF042h+labelsFF042a+labelsMM011h+labelsMM011a+labelsMM012h+labelsMM021h+labelsMM022h+labelsMM022a+
                labelsMM032h+labelsMM032a+labelsMM042h+labelsMM042a+labelsMM051h+labelsMM051a+labelsMM052a)

#total_labels.describe()
# Verify the combined dataset
print(f'Total audio files: {len(total_audio_files)}')
print(f'Total labels: {len(total_labels)}')


#Feature Extraction
def extract_features(audio_files):
    features = []
    for audio in audio_files:
        mfccs = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13)
        mfccs = np.mean(mfccs.T, axis=0)
        features.append(mfccs)
    return np.array(features)

# Extract features
features = extract_features(total_audio_files)


#Model training
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, total_labels, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

#save model
import joblib

# Save the model to a file
joblib.dump(model, 'random_forest_model5.pkl')
print("Model saved successfully!")

# Load the model from the file
loaded_model = joblib.load('random_forest_model5.pkl')

# Verify the loaded model
y_pred_loaded = loaded_model.predict(X_test)
accuracy_loaded = accuracy_score(y_test, y_pred_loaded)
print(f'Loaded model accuracy: {accuracy_loaded * 100:.2f}%')


# Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the Grid Search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit the Grid Search to the data
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(f'Best parameters: {best_params}')

# Train the model with the best parameters
#Fitting 5 folds for each of 108 candidates, totalling 540 fits
#Best parameters: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
#Accuracy after tuning: 89.60%
best_model = RandomForestClassifier(**best_params, random_state=42)
best_model.fit(X_train, y_train)
#Best parameters: {'max_depth': 30, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
# Evaluate the model
y_pred_best = best_model.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f'Accuracy after tuning: {accuracy_best * 100:.2f}%')

# Save the best_model to a file
joblib.dump(best_model, 'random_forest_bestmodel6.pkl')
print("best_model saved successfully!")


#NN model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier

def create_model(optimizer='adam', init='glorot_uniform'):
    model = Sequential()
    model.add(Dense(13, input_dim=13, kernel_initializer=init, activation='relu'))
    model.add(Dense(8, kernel_initializer=init, activation='relu'))
    model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

#Grid Search
from sklearn.model_selection import GridSearchCV

# Create the KerasClassifier# Create the KerasClassifier
model = KerasClassifier(model=create_model, verbose=0)

# Define the parameter grid
#param_grid = {
#    'model__optimizer': ['SGD', 'Adam'],  # Note the prefix "model__"
#    'model__init': ['glorot_uniform', 'normal', 'uniform'],  # Note the prefix "model__"
#    'batch_size': [10, 20, 40],
#    'epochs': [10, 50, 100]
#}


param_grid = {
    'model__optimizer': ['Adam'],  # Note the prefix "model__"
    'model__init': ['uniform'],  # Note the prefix "model__"
    'batch_size': [10],
    'epochs': [100]
}
# Initialize the Grid Search
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)

# Fit the Grid Search to the data
grid_result2 = grid.fit(X_train, y_train)

# Get the best parameters
best_params2 = grid_result2.best_params_
print(f'Best parameters: {best_params2}')
#Best parameters: {'batch_size': 10, 'epochs': 100, 'model__init': 'uniform', 'model__optimizer': 'Adam'}

# Create the best model using the best parameters


# Convert to NumPy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

best_model3 = create_model(optimizer=best_params2['model__optimizer'], init=best_params2['model__init'])

# Train the best model
best_model3.fit(X_train, y_train, epochs=best_params2['epochs'], batch_size=best_params2['batch_size'], verbose=1)

# Evaluate the model
loss, accuracy = best_model3.evaluate(X_test, y_test, verbose=0)
print(f'Accuracy after tuning: {accuracy * 100:.2f}%')
#Accuracy after tuning: 88.18%

#plot
history = best_model3.fit(X_train, y_train, epochs=best_params2['epochs'], batch_size=best_params2['batch_size'], verbose=1, validation_data=(X_test, y_test))
import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.show()

# Save the model to a file
best_model3.export('best_nn_model_tf')
print("Model saved successfully in TF2 SavedModel format!")

from tensorflow.keras.layers import TFSMLayer

# Load the model from the file
loaded_model = TFSMLayer('best_nn_model_tf', call_endpoint='serving_default')

import joblib

# Load the model from the file

# Verify the loaded model
import numpy as np
from sklearn.metrics import accuracy_score

# Make predictions with the loaded model
predictions = loaded_model(X_test)


# Extract the prediction values
pred_values = predictions['output_0'].numpy()

# Calculate loss (assuming mean squared error for regression)
loss = np.mean((pred_values - y_test) ** 2)

# Calculate accuracy (assuming binary classification)
accuracy = accuracy_score(np.round(pred_values), y_test)

print(f'Loaded model accuracy: {accuracy * 100:.2f}%')
print(f'Loaded model loss: {loss:.4f}')


#RANDOM FOREST
model = joblib.load('random_forest_bestmodel.pkl')

# Make predictions
prediction = model.predict(features)
probability = model.predict_proba(features)[0][1]

# Print the results
print(f'Prediction: {int(prediction[0])}')
print(f'Probability of dysarthria: {probability * 100:.2f}%')