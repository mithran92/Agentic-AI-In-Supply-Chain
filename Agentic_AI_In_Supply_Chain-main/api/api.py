
from fastapi import FastAPI
from agents.advanced_demand_agent import predict_demand_lstm

app = FastAPI()

@app.get("/predict")
def predict():
    return {"demand": predict_demand_lstm()}
