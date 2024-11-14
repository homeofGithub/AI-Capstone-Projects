import os
import sys
import pytest
import numpy as np
from flask import url_for
import joblib
import cv2  
from app_image import preprocess_image, app

# Fixtures for setup and teardown
@pytest.fixture
def client():
    # Setup the Flask test client
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def model():
    # Load the model
    model_path = 'models/image_model 1.pkl'
    if not os.path.exists(model_path):
        pytest.fail(f"Model file {model_path} not found.")
    return joblib.load(model_path)

@pytest.fixture
def sample_image(tmp_path):
    # Create a sample image for testing
    img_path = tmp_path / "test_image.jpg"
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255  # A white image
    cv2.imwrite(str(img_path), img)
    return str(img_path)

# Test Cases
def test_model_loading(model):
    """Test that the model loads correctly."""
    assert model is not None, "Model should not be None after loading."
    assert hasattr(model, "predict"), "Loaded model should have a predict method."

def test_preprocess_image(sample_image):
    """Test that preprocess_image correctly processes an image."""
    processed_image = preprocess_image(sample_image)
    
    # Check shape and normalization
    assert processed_image.shape == (1, 100, 100, 3), "Processed image should have shape (1, 100, 100, 3)."
    assert np.all(processed_image <= 1.0) and np.all(processed_image >= 0.0), "Processed image should be normalized to [0, 1]."

def test_upload_image(client, sample_image):
    """Test the /upload endpoint to ensure it processes an uploaded image correctly."""
    with open(sample_image, 'rb') as img_file:
        data = {'file': (img_file, 'test_image.jpg')}
        response = client.post('/upload', data=data, content_type='multipart/form-data')

    # Check if the response is successful
    assert response.status_code == 200, "The /upload endpoint should return status 200."
    assert b"Prediction:" in response.data, "The response should contain the prediction result."
    assert b"Probability of stroke:" in response.data, "The response should contain the probability result."
