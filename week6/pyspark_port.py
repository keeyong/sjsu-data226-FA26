# wget https://s3-geospatial.s3.us-west-2.amazonaws.com/trips.csv
from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.sql.functions import hour, count

conf = SparkConf()
conf.set("spark.app.name", "PySpark DataFrame #1")
conf.set("spark.master", "local[*]")

spark = SparkSession.builder\
        .config(conf=conf)\
        .getOrCreate()

df = spark.read.format("csv").option("header", "true").load("trips.csv")

df.printSchema()
df.show()
df.count()

df.rdd.getNumPartitions()

# Extract the hour of departure from the departure time column (e.g., `pickup_datetime`)
df = df.withColumn("hour", hour("tpep_pickup_datetime"))

# Group by hour and count trips
hourly_demand = df.groupBy("hour").agg(count("*").alias("trip_count"))

# Sort by trip_count to see peak hours
hourly_demand = hourly_demand.orderBy("trip_count", ascending=False)
hourly_demand.show()

df.createOrReplaceTempView("trips")

# SQL query to extract hour and count trips by hour
query = """
    SELECT hour(tpep_pickup_datetime) as hour, COUNT(*) as trip_count
    FROM trips
    GROUP BY 1
    ORDER BY trip_count DESC
"""

results = spark.sql(query)
results.show()

driver_results = results.collect()
# pyspark.sql.Row corresponds to a record in DataFrame
for r in driver_results:
    print(r)
