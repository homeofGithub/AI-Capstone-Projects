import os
import pytest
from app_image import app

# Define file paths for the test images
STROKE_IMAGE_PATH = os.path.join('tmp', 'aug_0_2.jpg')
NON_STROKE_IMAGE_PATH = os.path.join('tmp', 'aug_0_0.jpg')

@pytest.fixture
def client():
    """Fixture to create a test client for the Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_stroke_image(client):
    """Test uploading a stroke image."""
    with open(STROKE_IMAGE_PATH, 'rb') as stroke_image:
        # Simulate POST request to upload the image
        response = client.post('/upload', data={
            'file': (stroke_image, 'aug_0_2.jpg')
        }, content_type='multipart/form-data')
        
    # Assert that the response is successful
    assert response.status_code == 200

    # Parse the HTML response
    assert b"Prediction: 1" in response.data  # Stroke prediction is True
    assert b"Probability of stroke" in response.data  # Probability is displayed

def test_non_stroke_image(client):
    """Test uploading a non-stroke image."""
    with open(NON_STROKE_IMAGE_PATH, 'rb') as non_stroke_image:
        # Simulate POST request to upload the image
        response = client.post('/upload', data={
            'file': (non_stroke_image, 'aug_0_0.jpg')
        }, content_type='multipart/form-data')
        
    # Assert that the response is successful
    assert response.status_code == 200

    # Parse the HTML response
    assert b"Prediction: 0" in response.data  # Non-stroke prediction is False
    assert b"Probability of stroke" in response.data  # Probability is displayed
