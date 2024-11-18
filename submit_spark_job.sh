#!/bin/bash

# Exit on error
set -e

# Configuration
CLUSTER_NAME="gmm-cluster"
REGION="us-central1"
BUCKET_NAME="betterdata"

# Create a temporary directory for packaging
echo "Creating temporary directory..."
TEMP_DIR="temp_package"
mkdir -p $TEMP_DIR

# Copy all necessary files and directories
echo "Copying project files..."
cp -r src/ config/ main1.py requirements.txt $TEMP_DIR/

# Create __init__.py files if they don't exist
touch $TEMP_DIR/__init__.py
touch $TEMP_DIR/src/__init__.py
touch $TEMP_DIR/config/__init__.py

# Install dependencies to the temporary directory
echo "Installing dependencies..."
pip install -r requirements.txt --target $TEMP_DIR

# Create the zip file including all dependencies
echo "Creating zip file..."
cd $TEMP_DIR
zip -r ../job.zip ./*
cd ..

# Upload to GCS
echo "Uploading to Google Cloud Storage..."
gsutil cp job.zip gs://$BUCKET_NAME/
gsutil cp main1.py gs://$BUCKET_NAME/

# Submit the job
echo "Submitting Spark job..."
gcloud dataproc jobs submit pyspark \
    gs://$BUCKET_NAME/main1.py \
    --cluster=$CLUSTER_NAME \
    --region=$REGION \
    --py-files=gs://$BUCKET_NAME/job.zip \
    --jars=gs://spark-lib/bigquery/spark-bigquery-latest_2.12.jar \
    --properties="spark.executor.memory=4g,spark.driver.memory=4g,spark.executor.cores=4"

# Clean up temporary directory
echo "Cleaning up..."
rm -rf $TEMP_DIR

echo "Job submitted successfully!"