#!/bin/bash

# Exit on any error
set -e

# Configuration
PROJECT_ID="betterdata-441921"
REGION="us-central1"
CLUSTER_NAME="gmm-cluster-local"
BUCKET_NAME="betterdata"
DATASET_NAME="betterdata"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
    exit 1
}

# Ensure Cloud SDK is installed and configured
check_gcloud() {
    log "Checking gcloud configuration..."
    
    if ! command -v gcloud &> /dev/null; then
        error "gcloud not found. Please install Google Cloud SDK first."
    }
    
    # Set project and region
    gcloud config set project ${PROJECT_ID}
    gcloud config set compute/region ${REGION}
    
    # Verify authentication
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
        log "Please authenticate with gcloud..."
        gcloud auth login
    }
}

# Create temporary local directory
setup_workspace() {
    log "Setting up workspace..."
    
    WORKSPACE="./dataproc_workspace"
    mkdir -p ${WORKSPACE}
    
    # Copy required files to workspace
    cp -r ./src ${WORKSPACE}/
    cp -r ./config ${WORKSPACE}/
    cp main.py ${WORKSPACE}/
    cp requirements.txt ${WORKSPACE}/
    
    # Create a zip package
    (cd ${WORKSPACE} && zip -r ../gmm_package.zip .)
}

# Upload code to GCS
upload_to_gcs() {
    log "Uploading code to GCS..."
    
    # Create bucket if it doesn't exist
    if ! gsutil ls -b gs://${BUCKET_NAME} &>/dev/null; then
        gsutil mb -l ${REGION} gs://${BUCKET_NAME}
    }
    
    # Upload package
    gsutil cp gmm_package.zip gs://${BUCKET_NAME}/code/
    
    # Upload initialization script
    cat > ${WORKSPACE}/init_action.sh << 'EOF'
#!/bin/bash
pip install google-cloud-bigquery google-cloud-storage google-cloud-logging scikit-learn numpy pandas
EOF
    
    gsutil cp ${WORKSPACE}/init_action.sh gs://${BUCKET_NAME}/scripts/
}

# Create Dataproc cluster
create_cluster() {
    log "Creating Dataproc cluster: ${CLUSTER_NAME}"
    
    gcloud dataproc clusters create ${CLUSTER_NAME} \
        --region=${REGION} \
        --master-machine-type=n1-standard-8 \
        --worker-machine-type=n1-standard-8 \
        --num-workers=4 \
        --image-version=2.0-debian10 \
        --initialization-actions=gs://${BUCKET_NAME}/scripts/init_action.sh \
        --metadata="PIP_PACKAGES=google-cloud-bigquery google-cloud-storage google-cloud-logging scikit-learn" \
        --properties="spark:spark.executor.memory=8g,spark:spark.driver.memory=8g" \
        --optional-components=JUPYTER
}

# Submit job to cluster
submit_job() {
    log "Submitting Spark job to cluster..."
    
    JOB_ID="gmm-job-$(date +%Y%m%d-%H%M%S)"
    
    gcloud dataproc jobs submit pyspark \
        --cluster=${CLUSTER_NAME} \
        --region=${REGION} \
        --id=${JOB_ID} \
        gs://${BUCKET_NAME}/code/main.py \
        --jars=gs://spark-lib/bigquery/spark-bigquery-latest_2.12.jar \
        -- \
        --project_id=${PROJECT_ID} \
        --bucket_name=${BUCKET_NAME}
        
    echo ${JOB_ID} > job_id.txt
}

# Monitor job progress
monitor_job() {
    JOB_ID=$(cat job_id.txt)
    log "Monitoring job: ${JOB_ID}"
    
    while true; do
        STATUS=$(gcloud dataproc jobs describe ${JOB_ID} \
            --region=${REGION} \
            --format="value(status.state)")
            
        case ${STATUS} in
            "DONE")
                log "Job completed successfully!"
                break
                ;;
            "ERROR"|"CANCELLED")
                error "Job failed with status: ${STATUS}"
                break
                ;;
            *)
                log "Job status: ${STATUS}"
                sleep 30
                ;;
        esac
    done
}

# Cleanup resources
cleanup() {
    log "Cleaning up resources..."
    
    # Delete cluster
    gcloud dataproc clusters delete ${CLUSTER_NAME} \
        --region=${REGION} \
        --quiet
        
    # Clean local workspace
    rm -rf ${WORKSPACE} gmm_package.zip job_id.txt
}

# Main execution
main() {
    log "Starting local execution of GMM encoder on Dataproc..."
    
    check_gcloud
    setup_workspace
    upload_to_gcs
    create_cluster
    submit_job
    monitor_job
    
    read -p "Do you want to delete the cluster? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cleanup
    fi
    
    log "Execution completed!"
}

# Execute main function
main "$@"