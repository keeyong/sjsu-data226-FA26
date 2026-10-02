import pandas as pd
from datasets import load_dataset
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize Spark
spark = SparkSession.builder \
    .appName("Homework_Grading_Demo") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# Suppress noisy INFO logs so your print statements stand out
spark.sparkContext.setLogLevel("ERROR")

def log_step(step_num, title):
    print(f"\n{'='*20} STEP {step_num}: {title} {'='*20}")

# --- 1. Dataset Loading & Exploration ---
log_step(1, "Dataset Loading & Exploration")
dataset = load_dataset("imdb") # Change it for HW
df_pd = pd.DataFrame(dataset['train'])
spark_df = spark.createDataFrame(df_pd)

print(f"Check 1.1: Dataset Loaded. Total rows: {spark_df.count()}")
print("Check 1.2: Identifying text + label fields:")
spark_df.printSchema()
spark_df.show(5)

# --- 2. Data Preprocessing in Spark ---
log_step(2, "Data Preprocessing in Spark")
# Tokenization + Stopword Removal
tokenizer = Tokenizer(inputCol="text", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered")

# Feature extraction (TF-IDF)
hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=5000)
idf = IDF(inputCol="rawFeatures", outputCol="features")

print("Check 2.1: Pipeline stages defined (Tokenizer -> Remover -> TF-IDF)")
# (Note: Logic is applied during training in the Pipeline)

# --- 3. Model Training & Evaluation ---
log_step(3, "Model Training & Evaluation")
# Train/test split
train_data, test_data = spark_df.randomSplit([0.8, 0.2], seed=42)
print(f"Check 3.1: Split used. Train: {train_data.count()}, Test: {test_data.count()}")

# Classifier
lr = LogisticRegression(featuresCol="features", labelCol="label")
pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])

# Fit model
model = pipeline.fit(train_data)
print("Check 3.2: Model trained successfully.")

# Evaluation
predictions = model.transform(test_data)
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print(f"Check 3.3: Evaluation Metric (Accuracy): {accuracy:.4f}")

# --- 4. Model Saving & Inference ---
log_step(4, "Model Saving & Inference")
model_path = "spark_mlp_model"
model.write().overwrite().save(model_path)
print(f"Check 4.1: Saved trained pipeline to '{model_path}'")

# Load and run sample prediction
from pyspark.ml import PipelineModel
print(f"Load the saved model")
loaded_model = PipelineModel.load(model_path)
sample_text = spark.createDataFrame([("Brilliant and well-acted!",)], ["text"])
sample_pred = loaded_model.transform(sample_text)

print("Check 4.2: Inference on sample text:")
sample_pred.select("text", "prediction").show()

print("\n" + "*"*50)
print("DEMO LOGS COMPLETE")
print("*"*50)