# src/scripts/generate_data.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import expr, rand, col, monotonically_increasing_id
from pyspark.sql.types import StructType, StructField, DoubleType, LongType
import math
from src.utils.logging_utils import setup_logger
from typing import Optional

logger = setup_logger(__name__)

class LargeScaleDataGenerator:
    def __init__(
        self, 
        spark: SparkSession,
        target_rows: int = 1_000_000_000,  # 1 billion rows
        batch_size: int = 10_000_000,      # 10 million rows per batch
        num_partitions: Optional[int] = None
    ):
        self.spark = spark
        self.target_rows = target_rows
        self.batch_size = batch_size
        # Calculate optimal number of partitions if not provided
        self.num_partitions = num_partitions or self._calculate_optimal_partitions()
        
    def _calculate_optimal_partitions(self) -> int:
        """Calculate optimal number of partitions based on cluster resources"""
        conf = self.spark.sparkContext._conf.getAll()
        
        # Get executor config
        executor_cores = int(dict(conf).get("spark.executor.cores", "2"))
        num_executors = int(dict(conf).get("spark.executor.instances", "2"))
        
        # Aim for partitions to be ~128MB each
        target_partition_size = 128 * 1024 * 1024  # 128MB in bytes
        estimated_row_size = 20  # Assuming ~20 bytes per row
        total_size = self.target_rows * estimated_row_size
        
        # Calculate partitions based on data size and cluster resources
        size_based_partitions = math.ceil(total_size / target_partition_size)
        resource_based_partitions = executor_cores * num_executors * 2  # 2x cores for optimal parallelism
        
        return max(size_based_partitions, resource_based_partitions)

    def generate_schema(self) -> StructType:
        """Define the schema for generated data"""
        return StructType([
            StructField("id", LongType(), False),
            StructField("amount", DoubleType(), False)
        ])

    def generate_batch(self, batch_number: int) -> None:
        """Generate and save a batch of data"""
        try:
            # Calculate batch start ID
            start_id = batch_number * self.batch_size
            
            # Create batch DataFrame
            df = self.spark.range(
                start_id, 
                start_id + self.batch_size, 
                1, 
                self.num_partitions
            )
            
            # Add amount column with realistic distribution
            df = df.withColumn(
                "amount",
                expr("""
                    case 
                        when rand() < 0.7 then rand() * 1000  # 70% small transactions
                        when rand() < 0.9 then rand() * 5000  # 20% medium transactions
                        else rand() * 20000                   # 10% large transactions
                    end
                """)
            )
            
            # Write batch to BigQuery
            table = f"{self.project_id}.{self.dataset}.credit_data_{batch_number}"
            
            df.write \
                .format("bigquery") \
                .option("table", table) \
                .option("temporaryGcsBucket", self.temp_bucket) \
                .mode("overwrite") \
                .save()
            
            logger.info(f"Successfully generated and saved batch {batch_number}")
            
        except Exception as e:
            logger.error(f"Error generating batch {batch_number}: {str(e)}")
            raise

    def merge_batches(self) -> None:
        """Merge all batch tables into final table"""
        try:
            # Create merge query
            batch_tables = [f"credit_data_{i}" for i in range(math.ceil(self.target_rows / self.batch_size))]
            union_query = " UNION ALL ".join([
                f"SELECT * FROM `{self.project_id}.{self.dataset}.{table}`"
                for table in batch_tables
            ])
            
            final_query = f"""
            CREATE OR REPLACE TABLE `{self.project_id}.{self.dataset}.credit_data_final`
            AS {union_query}
            """
            
            # Execute merge query
            self.spark.sql(final_query)
            
            logger.info("Successfully merged all batches into final table")
            
        except Exception as e:
            logger.error(f"Error merging batches: {str(e)}")
            raise

    def cleanup_batch_tables(self) -> None:
        """Clean up intermediate batch tables"""
        try:
            batch_tables = [f"credit_data_{i}" for i in range(math.ceil(self.target_rows / self.batch_size))]
            
            for table in batch_tables:
                self.spark.sql(f"DROP TABLE IF EXISTS `{self.project_id}.{self.dataset}.{table}`")
            
            logger.info("Successfully cleaned up batch tables")
            
        except Exception as e:
            logger.error(f"Error cleaning up batch tables: {str(e)}")
            raise

    def generate_all_data(self) -> None:
        """Generate all data in batches and merge"""
        try:
            num_batches = math.ceil(self.target_rows / self.batch_size)
            
            logger.info(f"Starting data generation: {self.target_rows:,} rows in {num_batches} batches")
            
            # Generate batches
            for batch_num in range(num_batches):
                self.generate_batch(batch_num)
                logger.info(f"Completed batch {batch_num + 1}/{num_batches}")
            
            # Merge all batches
            self.merge_batches()
            
            # Cleanup
            self.cleanup_batch_tables()
            
            logger.info("Data generation completed successfully")
            
        except Exception as e:
            logger.error(f"Error in data generation: {str(e)}")
            raise