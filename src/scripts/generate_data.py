# src/scripts/generate_data.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import expr, rand, col
from pyspark.sql.types import StructType, StructField, DoubleType, LongType
import math
import pandas as pd
from src.utils.logging_utils import setup_logger
from typing import Optional, Dict, Any

logger = setup_logger(__name__)

class DataGenerator:
    def __init__(
        self,
        spark: SparkSession,
        gcs_bucket: str,
        sample_data_path: str,
        target_rows: int,
        batch_size: Optional[int] = None,
        num_partitions: Optional[int] = None
    ):
        self.spark = spark
        self.gcs_bucket = gcs_bucket
        self.sample_data_path = sample_data_path
        self.target_rows = target_rows
        self.batch_size = batch_size or min(10_000_000, target_rows // 10)
        self.num_partitions = num_partitions or self._calculate_optimal_partitions()
        self._analyze_sample_data()

    def _calculate_optimal_partitions(self) -> int:
        executor_cores = int(self.spark.conf.get("spark.executor.cores", "2"))
        num_executors = int(self.spark.conf.get("spark.executor.instances", "2"))
        
        target_partition_size = 128 * 1024 * 1024
        estimated_row_size = 20
        total_size = self.target_rows * estimated_row_size
        
        size_based_partitions = math.ceil(total_size / target_partition_size)
        resource_based_partitions = executor_cores * num_executors * 2
        
        return max(size_based_partitions, resource_based_partitions)

    def _analyze_sample_data(self) -> None:
        try:
            sample_df = pd.read_csv(self.sample_data_path)
            amount_col = sample_df["Amount"]
            
            self.stats = {
                "min": float(amount_col.min()),
                "max": float(amount_col.max()),
                "mean": float(amount_col.mean()),
                "std": float(amount_col.std()),
                "percentiles": {
                    p: float(amount_col.quantile(p/100))
                    for p in [25, 50, 75, 90, 95, 99]
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sample data: {str(e)}")
            raise

    def _generate_distribution_expr(self) -> str:
        stats = self.stats
        return f"""
        CASE 
            WHEN rand() < 0.25 THEN {stats['percentiles'][25]} + (rand() * ({stats['percentiles'][50]} - {stats['percentiles'][25]}))
            WHEN rand() < 0.75 THEN {stats['percentiles'][50]} + (rand() * ({stats['percentiles'][75]} - {stats['percentiles'][50]}))
            WHEN rand() < 0.90 THEN {stats['percentiles'][75]} + (rand() * ({stats['percentiles'][90]} - {stats['percentiles'][75]}))
            WHEN rand() < 0.95 THEN {stats['percentiles'][90]} + (rand() * ({stats['percentiles'][95]} - {stats['percentiles'][90]}))
            ELSE {stats['percentiles'][95]} + (rand() * ({stats['percentiles'][99]} - {stats['percentiles'][95]}))
        END
        """

    def generate_batch(self, batch_number: int, folder_name: str) -> None:
        try:
            start_id = batch_number * self.batch_size
            end_id = min(start_id + self.batch_size, self.target_rows)
            
            df = self.spark.range(start_id, end_id, 1, self.num_partitions)
            df = df.withColumn("amount", expr(self._generate_distribution_expr()))
            
            output_path = f"gs://{self.gcs_bucket}/{folder_name}/batch_{batch_number}"
            df.write.mode("overwrite").parquet(output_path)
                
            logger.info(f"Generated batch {batch_number} ({end_id - start_id:,} rows)")
            
        except Exception as e:
            logger.error(f"Error generating batch {batch_number}: {str(e)}")
            raise

    def generate_dataset(self, folder_name: str) -> None:
        try:
            logger.info(f"Starting generation of {self.target_rows:,} rows")
            num_batches = math.ceil(self.target_rows / self.batch_size)
            
            for batch in range(num_batches):
                self.generate_batch(batch, folder_name)
            
            final_path = f"gs://{self.gcs_bucket}/{folder_name}/final"
            
            self.spark.read.parquet(
                f"gs://{self.gcs_bucket}/{folder_name}/batch_*"
            ).write.mode("overwrite").parquet(final_path)
            
            final_count = self.spark.read.parquet(final_path).count()
            logger.info(f"Generated {final_count:,} rows")
            
            if final_count != self.target_rows:
                logger.warning(f"Expected {self.target_rows:,} rows but got {final_count:,}")
                
        except Exception as e:
            logger.error(f"Error in dataset generation: {str(e)}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        return self.stats