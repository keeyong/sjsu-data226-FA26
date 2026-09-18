
# Download 7 log.gz files in the same folder
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

spark = SparkSession.builder.appName("HandleLogFiles").getOrCreate()

# Load all .gz files in the directory into a DataFrame
df = spark.read.text("*.gz")

# Check the number of partitions
print(df.rdd.getNumPartitions())
df.show(truncate=False)

# Create a parsed dataframe (log_df)
# Extract the necessary information from log data using regular expressions
pattern = r'(\d+\.\d+\.\d+\.\d+) - - \[(.*?)\] "(.*?) (.*?) HTTP.*" (\d+) (\d+)'
log_df = df.select(
F.regexp_extract("value", pattern, 1).alias("ip"),
F.regexp_extract("value", pattern, 2).alias("timestamp"),
F.regexp_extract("value", pattern, 3).alias("method"),
F.regexp_extract("value", pattern, 4).alias("url"),
F.regexp_extract("value", pattern, 5).alias("status").cast("integer"),
F.regexp_extract("value", pattern, 6).alias("size").cast("integer")
)
log_df.show()

# Keep only 404 error logs
error_404_logs = log_df.filter(log_df.status == 404)

# Group by URL and then count, and sort by count in descending order
url_404_count = error_404_logs.groupBy("url").count().orderBy(F.desc("count"))

# print the outcome
url_404_count.show()

# Register the DataFrame as a temporary SQL table
log_df.createOrReplaceTempView("logs")

# Use SparkSQL to count URLs with 404 status
url_404_count = spark.sql("""
SELECT url, COUNT(1) as count
FROM logs
WHERE status = 404
GROUP BY url
ORDER BY count DESC
""")
url_404_count.show()
