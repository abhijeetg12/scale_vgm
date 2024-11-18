"""Data validation functions."""

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count, isnan, when
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

def validate_dataframe(df: DataFrame, required_columns: list):
    """
    Validate Spark DataFrame for required columns and data quality.
    
    Args:
        df: Spark DataFrame to validate
        required_columns: List of required column names
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    logger.info("Starting data validation")
    
    # Check required columns
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    
    # Check for nulls and invalid values
    for column in required_columns:
        null_count = df.filter(
            col(column).isNull() | 
            isnan(column) |
            (col(column) == '')
        ).count()
        
        if null_count > 0:
            logger.warning(f"Found {null_count} null values in column {column}")
    
    return True
