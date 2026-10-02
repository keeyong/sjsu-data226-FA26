from fastapi import FastAPI
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
import uvicorn

app = FastAPI(title="Spark ML Sentiment API")

# Initialize Spark and Load Model once on startup
spark = SparkSession.builder.appName("FastAPI_Spark").getOrCreate()
model = PipelineModel.load("spark_mlp_model")

@app.get("/")
def home():
    return {"message": "API is running. Send a POST request to /predict"}

@app.post("/predict")
async def predict(review: str):
    # Convert input string to Spark DataFrame
    input_df = spark.createDataFrame([(review,)], ["text"])
    
    # Run prediction
    prediction_df = model.transform(input_df)
    result = prediction_df.select("prediction").collect()[0][0]
    
    sentiment = "Positive" if result == 1.0 else "Negative"
    return {"review": review, "sentiment": sentiment}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)