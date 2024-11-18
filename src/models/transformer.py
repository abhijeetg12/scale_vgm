
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

class DataTransformer:
    """
    Modified transformer class for large-scale data processing.
    """
    
    def __init__(self, n_clusters=10, eps=0.005):
        """
        Initialize transformer.
        
        Args:
            n_clusters: Number of GMM clusters
            eps: Epsilon for component filtering
        """
        self.n_clusters = n_clusters
        self.eps = eps
        self.components = []
        self.ordering = []
        self.model = None
        
    def fit(self, data):
        """
        Fit the transformer on input data.
        
        Args:
            data: Input data array
        """
        logger.info("Fitting transformer")
        
        try:
            self.model = BayesianGaussianMixture(
                n_components=self.n_clusters,
                weight_concentration_prior_type='dirichlet_process',
                weight_concentration_prior=0.001,
                max_iter=100,
                n_init=1,
                random_state=42
            )
            
            self.model.fit(data.reshape(-1, 1))
            
            # Filter components
            old_comp = self.model.weights_ > self.eps
            mode_freq = np.unique(self.model.predict(data.reshape(-1, 1)))
            
            comp = []
            for i in range(self.n_clusters):
                if (i in mode_freq) and old_comp[i]:
                    comp.append(True)
                else:
                    comp.append(False)
                    
            self.components = comp
            
        except Exception as e:
            logger.error(f"Error in transformer fitting: {str(e)}")
            raise
            
    def transform(self, data):
        """
        Transform input data.
        
        Args:
            data: Input data array
            
        Returns:
            Transformed data array
        """
        logger.info("Transforming data")
        
        try:
            data = data.reshape(-1, 1)
            
            # Get model parameters
            means = self.model.means_.reshape((1, self.n_clusters))
            stds = np.sqrt(self.model.covariances_).reshape((1, self.n_clusters))
            
            # Normalize features
            features = (data - means) / (4 * stds)
            
            # Get component probabilities
            probs = self.model.predict_proba(data)
            probs = probs[:, self.components]
            
            # Store ordering for inverse transform
            col_sums = probs.sum(axis=0)
            self.ordering = np.argsort(-1 * col_sums)
            
            # Reorder probabilities
            reordered_probs = np.zeros_like(probs)
            for i, idx in enumerate(self.ordering):
                reordered_probs[:, i] = probs[:, idx]
                
            return np.hstack([features[:, 0:1], reordered_probs])
            
        except Exception as e:
            logger.error(f"Error in transformation: {str(e)}")
            raise