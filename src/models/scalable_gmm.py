# src/models/scalable_gmm.py
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import ArrayType, DoubleType, StructType, StructField
from sklearn.mixture import BayesianGaussianMixture
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

class ScalableGMMTransformer:
    """
    A scalable implementation of Gaussian Mixture Model for encoding large-scale data.
    """
    
    def __init__(
        self, 
        n_clusters: int = 10,
        eps: float = 0.005,
        batch_size: int = 1_000_000,
        sample_size: int = 100_000
    ):
        self.n_clusters = n_clusters
        self.eps = eps
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.model = None
        self.components = None
        self.ordering = None
        
    def fit(self, spark: SparkSession, df: DataFrame, column: str) -> None:
        """
        Fit GMM model using sampled data.
        """
        try:
            logger.info(f"Starting GMM fitting on {column}")
            
            # Sample data for fitting
            sample_df = df.select(column).sample(False, fraction=self.sample_size/df.count())
            sample_data = sample_df.toPandas()[column].values.reshape(-1, 1)
            
            # Fit BGM model
            self.model = BayesianGaussianMixture(
                n_components=self.n_clusters,
                weight_concentration_prior_type='dirichlet_process',
                weight_concentration_prior=0.001,
                max_iter=100,
                n_init=1,
                random_state=42
            )
            self.model.fit(sample_data)
            
            # Filter components
            self.components = (self.model.weights_ > self.eps)
            
            logger.info("GMM fitting completed successfully")
            
        except Exception as e:
            logger.error(f"Error in GMM fitting: {str(e)}")
            raise
            
    def transform_batch(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform a batch of data.
        """
        data = data.reshape(-1, 1)
        
        # Get model parameters
        means = self.model.means_.reshape((1, self.n_clusters))
        stds = np.sqrt(self.model.covariances_).reshape((1, self.n_clusters))
        
        # Normalize features
        features = (data - means) / (4 * stds)
        features = features[:, self.components]
        
        # Get probabilities
        probs = self.model.predict_proba(data)
        probs = probs[:, self.components]
        
        return np.clip(features[:, 0:1], -0.99, 0.99), probs
        
    def transform(self, spark: SparkSession, df: DataFrame, column: str) -> DataFrame:
        """
        Transform data using fitted GMM model.
        """
        try:
            logger.info(f"Starting transformation of column: {column}")
            
            # Define schema for transformed data
            n_components = np.sum(self.components)
            result_schema = ArrayType(DoubleType())
            
            # Create transform UDF
            @pandas_udf(result_schema)
            def transform_udf(series):
                features, probs = self.transform_batch(series.values)
                result = np.hstack([features, probs])
                return pd.Series([row.tolist() for row in result])
            
            # Apply transformation
            transformed_df = df.withColumn(
                "encoded",
                transform_udf(col(column))
            )
            
            logger.info("Transformation completed successfully")
            return transformed_df
            
        except Exception as e:
            logger.error(f"Error in transformation: {str(e)}")
            raise
            
    def inverse_transform_batch(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform a batch of encoded data.
        """
        features = data[:, 0:1]
        probs = data[:, 1:]
        
        # Get component means and stds
        means = self.model.means_.reshape(-1)[self.components]
        stds = np.sqrt(self.model.covariances_).reshape(-1)[self.components]
        
        # Get most likely component
        component_idx = np.argmax(probs, axis=1)
        
        # Inverse transform
        result = features * 4 * stds[component_idx].reshape(-1, 1) + means[component_idx].reshape(-1, 1)
        return result
        
    def inverse_transform(self, spark: SparkSession, df: DataFrame) -> DataFrame:
        """
        Inverse transform encoded data.
        """
        try:
            logger.info("Starting inverse transformation")
            
            @pandas_udf(DoubleType())
            def inverse_transform_udf(series):
                data = np.array([np.array(x) for x in series])
                result = self.inverse_transform_batch(data)
                return pd.Series(result.flatten())
            
            result_df = df.withColumn(
                "decoded",
                inverse_transform_udf(col("encoded"))
            )
            
            logger.info("Inverse transformation completed successfully")
            return result_df
            
        except Exception as e:
            logger.error(f"Error in inverse transformation: {str(e)}")
            raise