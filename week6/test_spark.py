from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .master("local[*]") \
    .appName("spark-3.5.8-test") \
    .getOrCreate()

spark.range(1, 6).show()
spark.stop()

