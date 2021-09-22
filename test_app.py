from fastapi.testclient import TestClient
from main import app
from datetime import datetime

# test to check the correct functioning of the /ping route
def test_ping():
    with TestClient(app) as client:
        response = client.get("/ping")
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"ping": "pong"}


# test to check if Iris Virginica is classified correctly
def test_pred_virginica():
    # defining a sample payload for the testcase
    payload = {
        "sepal_length": 3,
        "sepal_width": 5,
        "petal_length": 3.2,
        "petal_width": 4.4,
    }
    with TestClient(app) as client:
        response = client.post("/predict_flower", json=payload)
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"flower_class": "Iris Virginica"}

def test_appstatus():
    with TestClient(app) as client:
        response = client.get("/appstatus")
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"app": "running successfully","timestamp":datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}

def test_pred_setosa():
    # defining a sample payload for the testcase
    payload = {
        "sepal_length": 5.7,
        "sepal_width": 3.8,
        "petal_length": 1.7,
        "petal_width": 0.3
    }
    with TestClient(app) as client:
        response = client.post("/pred_setosa", json=payload)
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"flower_class": "Iris Setosa"}

def test_status():
    with TestClient(app) as client:
        response = client.get("/status")
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"status": "running"}

def test_pred_versicolor():
    # defining a sample payload for the testcase
    payload = {
        "sepal_length": 6.3,
        "sepal_width": 3.3,
        "petal_length": 4.7,
        "petal_width": 1.6
    }
    with TestClient(app) as client:
        response = client.post("/pred_versicolor", json=payload)
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"flower_class": "Iris Versicolour"}

def test_dummy():
    with TestClient(app) as client:
        response = client.get("/dummy")
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"dummy": "test"}