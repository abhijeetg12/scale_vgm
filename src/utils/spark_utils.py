# src/utils/spark_utils.py
from pyspark.sql import SparkSession
from config.spark_config import SPARK_CONF, SPARK_JARS

def create_spark_session(app_name: str = "GMM Encoder", 
                        memory_fraction: float = 0.8,
                        shuffle_partitions: int = 1000,
                        dynamic_allocation: bool = True) -> SparkSession:
    """
    Create Spark session with appropriate configuration.
    
    Args:
        app_name: Name of the Spark application
        memory_fraction: Fraction of memory to use for Spark
        shuffle_partitions: Number of shuffle partitions
        dynamic_allocation: Whether to enable dynamic allocation
        
    Returns:
        SparkSession instance
    """
    builder = (SparkSession.builder
               .appName(app_name))
    
    # Add memory configuration
    builder = builder.config("spark.memory.fraction", memory_fraction)
    builder = builder.config("spark.sql.shuffle.partitions", shuffle_partitions)
    
    # Add dynamic allocation if enabled
    if dynamic_allocation:
        builder = builder.config("spark.dynamicAllocation.enabled", "true")
        builder = builder.config("spark.shuffle.service.enabled", "true")
    
    # Add configurations from spark_config
    for key, value in SPARK_CONF.items():
        builder = builder.config(key, value)
    
    # Add required JARs
    if SPARK_JARS:
        builder = builder.config("spark.jars", ",".join(SPARK_JARS))
    
    return builder.getOrCreate()