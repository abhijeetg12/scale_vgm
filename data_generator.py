# %% [markdown]
# # Scalable GMM Encoder Demonstration
# This notebook demonstrates the usage of our production-ready GMM encoder implementation for processing 1B records.

# %%
# Import required libraries
import os
import sys
import time
from datetime import datetime
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Add project root to path
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import our production modules
from src.utils.spark_utils import create_spark_session
from src.scripts.generate_data import LargeScaleDataGenerator
from src.models.gmm_encoder import SparkGMMEncoder
from src.data.validation import validate_dataframe
from src.utils.logging_utils import setup_logger

logger = setup_logger("gmm_demo")

# %% [markdown]
# ## Initialize Spark Session with Optimized Configuration

# %%
def init_spark():
    """Initialize optimized Spark session for large-scale processing"""
    start_time = time.time()
    
    spark = create_spark_session(
        app_name="GMM Demo"
    )
    
    # Set optimized configurations for large-scale processing
    spark.conf.set("spark.sql.shuffle.partitions", "1000")  # Increased for 1B records
    spark.conf.set("spark.memory.fraction", "0.8")  # More memory for execution
    spark.conf.set("spark.memory.storageFraction", "0.3")  # Balance between storage/execution
    
    end_time = time.time()
    logger.info(f"Spark session initialized in {end_time - start_time:.2f} seconds")
    
    return spark

spark = init_spark()

# %% [markdown]
# ## Generate 1B Records Dataset
# We'll use our `LargeScaleDataGenerator` to create a realistic dataset based on the Credit.csv Amount column distribution.

# %%
def generate_large_dataset(spark):
    """Generate 1B records dataset"""
    start_time = time.time()
    
    generator = LargeScaleDataGenerator(
        spark=spark,
        target_rows=1_000_000_000,  # 1B rows
        batch_size=10_000_000       # Process in 10M row batches
    )
    
    generator.generate_all_data()
    
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Generated 1B records in {total_time:.2f} seconds ({total_time/3600:.2f} hours)")

# Generate the dataset
generate_large_dataset(spark)

# %% [markdown]
# ## Process Data with GMM Encoder
# We'll use our `SparkGMMEncoder` to process the data in chunks while maintaining state.

# %%
def process_with_gmm(spark):
    """Process the dataset using GMM encoder"""
    start_time = time.time()
    
    # Initialize encoder
    encoder = SparkGMMEncoder(n_clusters=10, eps=0.005)
    
    # Process in chunks of 100M rows
    chunk_size = 100_000_000
    total_chunks = 10
    
    # Training timing
    train_start = time.time()
    
    # Read first chunk for training
    first_chunk = spark.sql(f"""
        SELECT amount 
        FROM credit_data_final 
        LIMIT {chunk_size}
    """)
    
    # Fit encoder on first chunk
    encoder.fit(first_chunk, "amount")
    
    train_time = time.time() - train_start
    logger.info(f"GMM model trained in {train_time:.2f} seconds")
    
    # Process all chunks
    for chunk in range(total_chunks):
        chunk_start = time.time()
        
        # Read chunk
        chunk_df = spark.sql(f"""
            SELECT amount 
            FROM credit_data_final 
            WHERE id >= {chunk * chunk_size} 
            AND id < {(chunk + 1) * chunk_size}
        """)
        
        # Validate data
        if not validate_dataframe(chunk_df, ["amount"]):
            raise ValueError(f"Data validation failed for chunk {chunk}")
        
        # Transform chunk
        encoded_df = encoder.transform(chunk_df, "amount")
        
        # Save transformed chunk
        encoded_df.write.format("parquet") \
            .mode("overwrite") \
            .save(f"encoded_data/chunk_{chunk}")
        
        chunk_time = time.time() - chunk_start
        logger.info(f"Processed chunk {chunk + 1}/{total_chunks} in {chunk_time:.2f} seconds")
    
    total_time = time.time() - start_time
    logger.info(f"Total processing time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
    
    return total_time

# Process the dataset
total_processing_time = process_with_gmm(spark)

# %% [markdown]
# ## Verify Results
# Let's verify the quality of our GMM encoding by checking a sample of the results.

# %%
def verify_results(spark):
    """Verify encoding quality on a sample"""
    # Read sample from first chunk
    sample_df = spark.read.parquet("encoded_data/chunk_0").limit(1000)
    
    # Convert to pandas for analysis
    sample_pd = sample_df.toPandas()
    
    # Check encoded values distribution
    encoded_stats = sample_pd.encoded.apply(lambda x: x[0]).describe()
    print("\nEncoded Values Statistics:")
    print(encoded_stats)
    
    # Check component probabilities
    comp_probs = sample_pd.encoded.apply(lambda x: x[1:]).apply(np.max)
    print("\nComponent Assignment Confidence:")
    print(comp_probs.describe())

verify_results(spark)

# %% [markdown]
# ## Performance Summary
# 
# The implementation successfully processed 1B records with the following performance characteristics:
# 
# 1. Data Generation: Created 1B realistic records based on Credit.csv distribution
# 2. GMM Training: Trained on first 100M records for optimal performance
# 3. Batch Processing: Processed data in 10 chunks of 100M records each
# 4. Storage: Used Parquet format for efficient storage and retrieval
# 
# Key optimizations used:
# - Optimized Spark configuration for large-scale processing
# - Efficient batch processing to handle memory constraints
-# Validation at each step to ensure data quality
# - Parallel processing using Spark's distributed computing capabilities

# Stop Spark session
spark.stop()