from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .master("local[*]") \
    .appName("spark-3.5.8-test") \
    .getOrCreate()

spark.range(1, 6).show()  # Creates a DataFrame containing values from 1 to 5 and then print
spark.stop()

