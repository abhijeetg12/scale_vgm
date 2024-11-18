# main.py
import os
import sys
import time
from typing import Optional
from datetime import datetime

# Add the current directory to Python path
current_dir = os.path.abspath(os.path.dirname(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from config.gcp_config import PROJECT_ID, BUCKET_NAME, BIGQUERY_DATASET
from src.utils.spark_utils import create_spark_session
from src.data.ingestion import read_from_bigquery
from src.data.validation import validate_dataframe
from src.models.gmm_encoder import SparkGMMEncoder
from src.scripts.generate_data import LargeScaleDataGenerator
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

def log_timing(func):
    """Decorator to log function execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"Starting {func.__name__} at {datetime.now()}")
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Completed {func.__name__} in {duration:.2f} seconds")
        return result
    return wrapper

@log_timing
def generate_data(spark, rows: Optional[int] = None) -> None:
    """Generate large-scale dataset based on Credit.csv Amount distribution"""
    generator = LargeScaleDataGenerator(
        spark=spark,
        target_rows=rows or 1_000_000_000,  # Default to 1B rows
        batch_size=10_000_000,  # 10M rows per batch
    )
    generator.generate_all_data()

@log_timing
def process_chunk(spark, chunk_id: int, chunk_size: int, encoder: SparkGMMEncoder) -> None:
    """Process a single chunk of data"""
    chunk_start = time.time()
    
    # Read chunk using optimized query
    chunk_query = f"""
    SELECT amount 
    FROM `{PROJECT_ID}.{BIGQUERY_DATASET}.credit_data_final`
    WHERE MOD(id, {chunk_size}) >= {chunk_id * chunk_size}
    AND MOD(id, {chunk_size}) < {(chunk_id + 1) * chunk_size}
    """
    
    df = read_from_bigquery(spark, query=chunk_query)
    
    # Validate data
    if not validate_dataframe(df, ["amount"]):
        raise ValueError(f"Data validation failed for chunk {chunk_id}")
    
    # Transform chunk
    encoded_df = encoder.transform(df, "amount")
    
    # Save chunk results with optimized write
    chunk_table = f"encoded_data_chunk_{chunk_id}"
    encoded_df.write.format("bigquery") \
        .option("temporaryGcsBucket", BUCKET_NAME) \
        .option("table", f"{PROJECT_ID}.{BIGQUERY_DATASET}.{chunk_table}") \
        .mode("overwrite") \
        .save()
    
    chunk_duration = time.time() - chunk_start
    logger.info(f"Processed chunk {chunk_id} in {chunk_duration:.2f} seconds")

def main():
    total_start_time = time.time()
    logger.info("Starting GMM encoding process for 1B records")
    
    try:
        # Initialize Spark with optimized configuration for large-scale processing
        spark = create_spark_session(
            app_name="GMM_Encoder_1B",
            memory_fraction=0.8,
            shuffle_partitions=2000,  # Increased for better parallelism
            dynamic_allocation=True
        )
        
        # Configure temporary bucket for BigQuery operations
        spark.conf.set('temporaryGcsBucket', BUCKET_NAME)
        
        # Check if we need to generate 1B records
        logger.info("Checking existing data volume")
        count_query = f"""
        SELECT COUNT(*) as count 
        FROM `{PROJECT_ID}.{BIGQUERY_DATASET}.credit_data_final`
        """
        
        try:
            existing_count = spark.sql(count_query).first()['count']
            if existing_count < 1_000_000_000:
                logger.info("Generating 1B records dataset")
                generate_data(spark)
        except Exception:
            logger.info("No existing data found. Generating 1B records dataset")
            generate_data(spark)
        
        # Process data in optimized chunks
        chunk_size = 50_000_000  # Process 50M rows at a time
        total_chunks = 20  # For 1B rows
        
        # Initialize encoder with sample data
        logger.info("Initializing GMM encoder")
        sample_query = f"""
        SELECT amount 
        FROM `{PROJECT_ID}.{BIGQUERY_DATASET}.credit_data_final`
        TABLESAMPLE SYSTEM (0.1 PERCENT)  -- Sample 0.1% of data for fitting
        """
        sample_df = read_from_bigquery(spark, query=sample_query)
        
        encoder_start = time.time()
        encoder = SparkGMMEncoder(n_clusters=10)
        encoder.fit(sample_df, "amount")
        logger.info(f"Encoder initialization completed in {time.time() - encoder_start:.2f} seconds")
        
        # Process chunks in parallel
        for chunk in range(total_chunks):
            process_chunk(spark, chunk, chunk_size, encoder)
        
        # Merge results efficiently
        logger.info("Merging final results")
        merge_start = time.time()
        merge_query = f"""
        CREATE OR REPLACE TABLE `{PROJECT_ID}.{BIGQUERY_DATASET}.encoded_data_final` AS
        SELECT * FROM `{PROJECT_ID}.{BIGQUERY_DATASET}.encoded_data_chunk_*`
        """
        spark.sql(merge_query)
        logger.info(f"Merge completed in {time.time() - merge_start:.2f} seconds")
        
        # Cleanup
        cleanup_start = time.time()
        logger.info("Cleaning up temporary tables")
        for chunk in range(total_chunks):
            spark.sql(f"DROP TABLE IF EXISTS `{PROJECT_ID}.{BIGQUERY_DATASET}.encoded_data_chunk_{chunk}`")
        logger.info(f"Cleanup completed in {time.time() - cleanup_start:.2f} seconds")
        
        total_duration = time.time() - total_start_time
        logger.info(f"Total processing completed in {total_duration:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise
    finally:
        if 'spark' in locals():
            spark.stop()

if __name__ == "__main__":
    main()