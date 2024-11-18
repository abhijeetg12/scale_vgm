"""GCP configuration settings."""

# GCP Project settings
PROJECT_ID = "betterdata-441921"  # Your actual project ID
REGION = "us-central1"

# Storage settings
BUCKET_NAME = "betterdata"
BIGQUERY_DATASET = "betterdata"
BIGQUERY_TABLE = "credit_data"

# Dataproc settings
CLUSTER_NAME = "gmm-cluster"
CLUSTER_CONFIG = {
    "master_config": {
        "machine_type": "n1-standard-4",
    },
    "worker_config": {
        "machine_type": "n1-standard-4",
        "num_instances": 2,
    }
}

