#!/bin/bash

# Exit on error
set -e

# Install Python dependencies
pip install -r requirements.txt

# Set up GCP credentials
if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "Please set GOOGLE_APPLICATION_CREDENTIALS environment variable"
    exit 1
fi

# Create GCS bucket if it doesn't exist
gsutil mb -p $PROJECT_ID gs://$BUCKET_NAME || true

# Create BigQuery dataset if it doesn't exist
bq mk --dataset $PROJECT_ID:$BIGQUERY_DATASET || true

echo "Environment setup complete!"
