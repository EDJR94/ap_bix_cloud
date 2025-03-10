#!/bin/bash

# Set your project ID
PROJECT_ID="mlos-udemy"

# Enable required APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable run.googleapis.com

# Create Artifact Registry repository
gcloud artifacts repositories create ml-models \
    --repository-format=docker \
    --location=us-central1 \
    --description="ML model containers"

# Create Cloud Build trigger
gcloud builds triggers create github \
    --name=iris-ml-model-trigger \
    --repository=EDJR94/ap_bix_cloud\
    --branch-pattern=main \
    --build-config=cloudbuild.yaml

# Grant Cloud Run deployer role to Cloud Build service account
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')
SERVICE_ACCOUNT="${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/run.admin"

gcloud iam service-accounts add-iam-policy-binding \
    $PROJECT_NUMBER-compute@developer.gserviceaccount.com \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/iam.serviceAccountUser"