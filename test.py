import requests
import json

# Replace with your actual Cloud Run URL
url = "https://iris-predictor-273533929243.us-central1.run.app/predict"

# Sample data (Iris setosa features: sepal length, sepal width, petal length, petal width)
data = {
    "features": [5.1, 3.5, 1.4, 0.2]  # Iris setosa example
}

# Send prediction request
response = requests.post(url, json=data)

# Display results
result = response.json()
print(result)