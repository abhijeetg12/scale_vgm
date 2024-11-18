# src/scripts/generate_data.py
import numpy as np
import pandas as pd
from google.cloud import bigquery
from google.cloud import storage
import io
import math
from src.utils.logging_utils import setup_logger
from config.gcp_config import PROJECT_ID, BUCKET_NAME, BIGQUERY_DATASET

logger = setup_logger(__name__)

class CreditDataGenerator:
    def __init__(self):
        self.storage_client = storage.Client()
        self.bigquery_client = bigquery.Client()
        self.bucket = self.storage_client.get_bucket(BUCKET_NAME)
        
    def generate_batch(self, batch_size: int, batch_number: int) -> pd.DataFrame:
        """
        Generate credit data following the original distribution pattern
        """
        np.random.seed(42 + batch_number)  # Ensure reproducibility while having different batches
        
        # Generate base amounts using lognormal distribution
        amounts = np.random.lognormal(mean=6, sigma=0.8, size=batch_size)
        
        # Create multiple modes to match original data characteristics
        modes = np.random.choice(3, size=batch_size, p=[0.7, 0.2, 0.1])
        
        # Adjust amounts based on modes
        amounts[modes == 1] *= 2  # Medium transactions
        amounts[modes == 2] *= 5  # Large transactions
        
        # Add some random noise
        noise = np.random.normal(0, amounts * 0.05)  # 5% noise
        amounts += noise
        
        # Ensure amounts are positive and round to 2 decimals
        amounts = np.maximum(amounts, 0)
        amounts = np.round(amounts, 2)
        
        return pd.DataFrame({'Amount': amounts})
    
    def upload_batch_to_gcs(self, df: pd.DataFrame, batch_num: int):
        """Upload batch to GCS"""
        blob_name = f'data/credit_data/batch_{batch_num:05d}.csv'
        blob = self.bucket.blob(blob_name)
        
        # Convert to CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        
        # Upload to GCS
        blob.upload_from_string(csv_buffer.getvalue())
        logger.info(f"Uploaded batch {batch_num} to GCS")
        
    def load_to_bigquery(self):
        """Load all data from GCS to BigQuery"""
        # Create dataset if it doesn't exist
        dataset_ref = self.bigquery_client.dataset(BIGQUERY_DATASET)
        try:
            self.bigquery_client.get_dataset(dataset_ref)
        except Exception:
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = "US"
            self.bigquery_client.create_dataset(dataset)
        
        # Create table
        table_id = f"{PROJECT_ID}.{BIGQUERY_DATASET}.credit_data"
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.CSV,
            skip_leading_rows=1,
            autodetect=True,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE
        )
        
        uri = f"gs://{BUCKET_NAME}/data/credit_data/*.csv"
        load_job = self.bigquery_client.load_table_from_uri(
            uri, table_id, job_config=job_config
        )
        load_job.result()  # Wait for job to complete
        
        logger.info("Data loaded to BigQuery successfully")

def main():
    # Configuration
    TOTAL_RECORDS = 1_000_000_000  # 1 billion records
    BATCH_SIZE = 1_000_000  # 1 million records per batch
    
    generator = CreditDataGenerator()
    num_batches = math.ceil(TOTAL_RECORDS / BATCH_SIZE)
    
    logger.info(f"Starting data generation: {TOTAL_RECORDS:,} records in {num_batches:,} batches")
    
    try:
        # Generate and upload data in batches
        for batch_num in range(num_batches):
            logger.info(f"Processing batch {batch_num + 1}/{num_batches}")
            
            # Calculate actual batch size (last batch might be smaller)
            current_batch_size = min(BATCH_SIZE, 
                                   TOTAL_RECORDS - (batch_num * BATCH_SIZE))
            
            # Generate batch
            df_batch = generator.generate_batch(current_batch_size, batch_num)
            
            # Upload to GCS
            generator.upload_batch_to_gcs(df_batch, batch_num)
            
            if (batch_num + 1) % 10 == 0:
                logger.info(f"Completed {(batch_num + 1) * BATCH_SIZE:,} records")
        
        # Load all data to BigQuery
        logger.info("Loading data to BigQuery...")
        generator.load_to_bigquery()
        
        logger.info("Data generation completed successfully")
        
    except Exception as e:
        logger.error(f"Error in data generation: {str(e)}")
        raise

if __name__ == "__main__":
    main()