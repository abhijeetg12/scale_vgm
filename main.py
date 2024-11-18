# main.py
import os
import sys

# Add the current directory to Python path
current_dir = os.path.abspath(os.path.dirname(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from config.gcp_config import PROJECT_ID, BUCKET_NAME, BIGQUERY_DATASET
from config.spark_config import SPARK_CONF, SPARK_JARS
from src.utils.spark_utils import create_spark_session
from src.data.ingestion import read_from_bigquery
from src.data.validation import validate_dataframe
from src.models.gmm_encoder import SparkGMMEncoder
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

def main():
    try:
        # Initialize Spark
        logger.info("Initializing Spark session")
        spark = create_spark_session()  # Remove all arguments here
        
        # Configure temporary bucket for BigQuery
        spark.conf.set('temporaryGcsBucket', BUCKET_NAME)
        
        # Read data
        logger.info("Reading data from BigQuery")
        df = read_from_bigquery(spark)
        
        # Validate data
        logger.info("Validating data")
        if not validate_dataframe(df, ["Amount"]):
            raise ValueError("Data validation failed")
        
        # Initialize and fit encoder
        logger.info("Fitting GMM encoder")
        encoder = SparkGMMEncoder(n_clusters=10)
        encoder.fit(df, "Amount")
        
        # Transform data
        logger.info("Transforming data")
        encoded_df = encoder.transform(df, "Amount")
        
        # Save results
        logger.info("Saving results to BigQuery")
        encoded_df.write.format("bigquery") \
            .option("temporaryGcsBucket", BUCKET_NAME) \
            .option("table", f"{PROJECT_ID}.{BIGQUERY_DATASET}.encoded_data") \
            .mode("overwrite") \
            .save()
        
        logger.info("Processing completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise
    finally:
        if 'spark' in locals():
            spark.stop()

if __name__ == "__main__":
    main()