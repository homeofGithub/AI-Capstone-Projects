import pytest
from app import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_index(client):
    # Send a GET request to the '/' route
    response = client.get('/')

    # Assert that the response status code is 200 (OK)
    assert response.status_code == 200

    # Check if the rendered HTML contains specific content from index.html
    assert b"Upload Audio File for Dysarthria Detection" in response.data  # Check for the <h2> content

