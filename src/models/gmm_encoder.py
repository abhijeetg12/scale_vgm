# src/models/gmm_encoder.py
from pyspark.sql import DataFrame
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import udf, pandas_udf, col
from pyspark.sql.types import ArrayType, DoubleType
import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

class SparkGMMEncoder:
    def __init__(self, n_clusters: int = 10, eps: float = 0.005):
        self.n_clusters = n_clusters
        self.eps = eps
        self.model = None
        
    def fit(self, spark_df: DataFrame, input_col: str):
        logger.info("Fitting GMM model")
        
        # Sample data for fitting
        sample_size = min(100000, spark_df.count())
        sample_data = spark_df.select(input_col).sample(False, sample_size/spark_df.count()).toPandas()
        
        # Fit BGM
        self.model = BayesianGaussianMixture(
            n_components=self.n_clusters,
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=0.001,
            max_iter=100,
            n_init=1,
            random_state=42
        )
        
        self.model.fit(sample_data[input_col].values.reshape(-1, 1))
        
    def transform(self, spark_df: DataFrame, input_col: str):
        logger.info("Transforming data using GMM")
        
        # Create transform UDF
        @pandas_udf(ArrayType(DoubleType()))
        def transform_udf(series):
            # Convert input to numpy array
            values = series.values.reshape(-1, 1)
            
            # Get means and stds
            means = self.model.means_.reshape(1, self.n_clusters)
            stds = np.sqrt(self.model.covariances_).reshape(1, self.n_clusters)
            
            # Normalize values
            features = (values - means) / (4 * stds)
            
            # Get component probabilities
            probs = self.model.predict_proba(values)
            
            # Combine features and probabilities
            result = np.hstack([features[:, 0:1], probs])
            
            # Convert numpy array to pandas Series of arrays
            return pd.Series([row for row in result])
        
        return spark_df.withColumn("encoded", transform_udf(col(input_col)))