FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train the model during container build
RUN python -c "from main import train_model; train_model()"

EXPOSE 8080

# Run the FastAPI app with Uvicorn
CMD ["python", "main.py"]