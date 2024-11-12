import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from app import extract_mfcc  # Replace with the actual import

# Mock data for testing
mock_audio = np.random.rand(16000)  # Mocked 1-second audio with 16kHz sampling rate
mock_sr = 16000  # Standard sample rate
mock_mfccs = np.random.rand(13, 100)  # Mocked MFCCs (13 MFCCs, 100 frames)


# Test case for extract_mfcc function
@patch("librosa.load")
@patch("librosa.feature.mfcc")
def test_extract_mfcc(mock_mfcc, mock_load):
    # Mock the return values of librosa.load and librosa.feature.mfcc
    mock_load.return_value = (mock_audio, mock_sr)
    mock_mfcc.return_value = mock_mfccs

    # Call the function with a dummy file path
    audio_path = "dummy/path/to/audio.wav"
    result = extract_mfcc(audio_path)

    # Check that librosa.load was called with the correct parameters
    mock_load.assert_called_once_with(audio_path, sr=16000)

    # Check that librosa.feature.mfcc was called with the expected parameters
    mock_mfcc.assert_called_once_with(y=mock_audio, sr=mock_sr, n_mfcc=13)

    # Check the result's shape
    assert result.shape == (1, 13), f"Expected shape (1, 13), but got {result.shape}"

    # Optionally: Check that the result is a 2D numpy array with the correct values
    assert isinstance(result, np.ndarray), "Result should be a numpy array"
    assert result.ndim == 2, "Result should be a 2D array"
    assert result.shape[1] == 13, "Expected 13 MFCCs per sample"

    # Check that the mean of MFCCs is computed (optional, depends on your expected output)
    mean_mfcc = np.mean(mock_mfccs.T, axis=0)
    assert np.allclose(result, mean_mfcc.reshape(1, -1)), "MFCCs do not match the expected mean values"
