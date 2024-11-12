import os
import pytest
from flask import Flask, request, render_template
from io import BytesIO
from unittest.mock import patch, MagicMock

# Assuming your app is named 'app' in a file called 'app.py'
from app import app, extract_mfcc, model


@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

@patch('app.extract_mfcc')
@patch('app.model.predict')
@patch('app.model.predict_proba')
def test_upload_valid_file(mock_predict_proba, mock_predict, mock_extract_mfcc, client):
    # Mocking the external methods
    mock_extract_mfcc.return_value = [0.1, 0.2, 0.3]  # Mock the output of extract_mfcc
    mock_predict.return_value = [1]  # Mock the prediction (e.g., class 1)
    mock_predict_proba.return_value = [[0.1, 0.9]]  # Mock the predicted probability (90%)

    # Create a mock file in memory (valid WAV file)
    data = {
        'file': (BytesIO(b"fake wav file content"), 'test.wav')
    }

    # Perform a POST request to upload the file
    response = client.post('/upload', data=data, content_type='multipart/form-data')

    # Debugging: Print the response content
    print(response.data.decode())

    # Assert that the file was processed correctly and the result is rendered
    assert response.status_code == 200
    assert b'Prediction: 1' in response.data  # Check that the correct prediction is rendered
    assert b'Probability of dysarthria: 90.0%' in response.data