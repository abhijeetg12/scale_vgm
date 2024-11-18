"""Spark configuration settings."""

SPARK_CONF = {
    "spark.executor.memory": "4g",
    "spark.driver.memory": "4g",
    "spark.executor.cores": "2",
    "spark.driver.cores": "2",
    "spark.sql.shuffle.partitions": "100",
    "spark.default.parallelism": "100"
}

SPARK_JARS = ["gs://spark-lib/bigquery/spark-bigquery-latest_2.12.jar"]
