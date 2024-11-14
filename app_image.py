from flask import Flask, request, render_template
import joblib
import numpy as np
import cv2
import os
import tensorflow as tf

# Flask instantiation
app = Flask(__name__)

# Ensure the tmp directory exists
# Uploaded images will be stored here
if not os.path.exists('tmp'):
    os.makedirs('tmp')

# Load the saved model
model = joblib.load('models/image_model 1.pkl')

# Preprocess image
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
    return render_template('index_image.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']

    if file and file.filename.endswith('.jpg'):

        # Save the image to a temporary location
        filepath = os.path.join('tmp', file.filename)
        file.save(filepath)

        # Preprocess the image
        image = preprocess_image(filepath)

        # Make prediction
        prediction = model.predict(image)

        # Predicted class is the element with the highest value
        predicted_class_index = tf.argmax(prediction, axis=1)
        predicted_class_index = predicted_class_index.numpy()[0]
        
        # Sample prediction result: [0.9961105  0.00388956]
        # Index with the highest probability is row 0, column 0
        probability = prediction[0, predicted_class_index] # row number 0, column number predict_class_index

        print(predicted_class_index)
        print(probability)

        if (predicted_class_index == 0):
            probability = 1 - probability        

        result = {
            'prediction': predicted_class_index,
            'probability': round(probability * 100, 2)
        }

        return render_template('index_image.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
