# main.py
import os
import pickle
import numpy as np
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

app = FastAPI(title="Iris Classifier API", description="A simple ML model API using FastAPI")

# Define request model with validation
class PredictionRequest(BaseModel):
    features: List[float]

# Define response model
class PredictionResponse(BaseModel):
    prediction: int
    class_name: str
    probability: float

# Train a simple model (in production, you'd load a pre-trained model)
def train_model():
    print("Training model...")
    # Load the Iris dataset as an example
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Train a random forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save the model
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Model trained and saved.")
    return model

# Load or train the model on startup
if os.path.exists('model.pkl'):
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
else:
    model = train_model()

# Load iris dataset for class names
iris = load_iris()

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    # Check input dimensions
    if len(request.features) != 4:
        raise HTTPException(status_code=400, detail="Iris model expects 4 features")
    
    # Make prediction
    features = np.array(request.features).reshape(1, -1)
    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    
    # Get the probability for the predicted class
    probability = float(probabilities[prediction])
    
    # Map prediction to class name
    class_name = iris.target_names[prediction]
    
    return {
        "prediction": prediction,
        "class_name": class_name,
        "probability": probability
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)