"""Data ingestion functions."""

from google.cloud import bigquery
from pyspark.sql import SparkSession
from config.gcp_config import PROJECT_ID, BIGQUERY_DATASET, BIGQUERY_TABLE
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

def create_bigquery_client():
    """Create BigQuery client."""
    return bigquery.Client()

def read_from_bigquery(spark: SparkSession, query: str = None):
    """
    Read data from BigQuery using Spark.
    
    Args:
        spark: SparkSession instance
        query: Optional query string
    
    Returns:
        DataFrame: Spark DataFrame containing the query results
    """
    logger.info("Starting BigQuery data ingestion")
    
    try:
        if query:
            return (spark.read.format("bigquery")
                   .option("query", query)
                   .load())
        else:
            return (spark.read.format("bigquery")
                   .option("table", f"{PROJECT_ID}.{BIGQUERY_DATASET}.{BIGQUERY_TABLE}")
                   .load())
    except Exception as e:
        logger.error(f"Error reading from BigQuery: {str(e)}")
        raise