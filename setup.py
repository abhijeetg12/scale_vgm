from setuptools import setup, find_packages

setup(
    name="cloud-gmm-encoder",
    version="0.1.0",
    packages=find_packages(),
    package_dir={'': 'src'},  # Add this line to specify src directory
    install_requires=[
        "google-cloud-bigquery>=3.11.4",
        "google-cloud-storage>=2.10.0",
        "google-cloud-logging>=3.5.0",
        "pyspark>=3.4.1",
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "scikit-learn>=1.3.0",
        "pytest>=7.4.0",
    ],
    python_requires=">=3.8",
)