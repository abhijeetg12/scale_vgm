#!/bin/bash

# Exit on error
set -e

# Load configuration
source config/gcp_config.py

# Create a temporary directory for packaging
echo "Preparing package..."
TEMP_DIR=$(mktemp -d)
cp -r src config main.py requirements.txt setup.py "$TEMP_DIR/"
cd "$TEMP_DIR"

# Install package in development mode
pip install -e .

# Create zip file of the package
echo "Creating deployment package..."
zip -r cloud_gmm_encoder.zip ./*

# Copy to GCS
echo "Copying package to GCS..."
gsutil cp cloud_gmm_encoder.zip gs://$BUCKET_NAME/packages/

# Create Dataproc cluster
echo "Creating Dataproc cluster..."
gcloud dataproc clusters create $CLUSTER_NAME \
    --region=$REGION \
    --master-machine-type=n1-standard-4 \
    --worker-machine-type=n1-standard-4 \
    --num-workers=2 \
    --image-version=2.0-debian10 \
    --initialization-actions=gs://goog-dataproc-initialization-actions-${REGION}/python/pip-install.sh \
    --metadata="PIP_PACKAGES=google-cloud-bigquery google-cloud-storage google-cloud-logging scikit-learn"

# Submit Spark job
echo "Submitting Spark job..."
gcloud dataproc jobs submit pyspark \
    --cluster=$CLUSTER_NAME \
    --region=$REGION \
    --py-files=gs://$BUCKET_NAME/packages/cloud_gmm_encoder.zip \
    main.py

# Clean up temporary directory
cd -
rm -rf "$TEMP_DIR"

echo "Deployment complete!"