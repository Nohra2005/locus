# LOCUS: main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from prometheus_fastapi_instrumentator import Instrumentator
from ranker import LocusRanker

app = FastAPI()
Instrumentator().instrument(app).expose(app)
ranker = LocusRanker()

# Define the data format we expect to receive
class RankRequest(BaseModel):
    query_vector: List[float]
    candidate_vectors: List[List[float]]

@app.get("/")
def read_root():
    return {"status": "online", "service": "Locus Ranking Engine"}

@app.post("/rank")
def rank_vectors(payload: RankRequest):
    # 1. Get data from the request
    query = payload.query_vector
    candidates = payload.candidate_vectors
    
    # 2. Run the math logic
    results = ranker.predict(query, candidates)
    
    return {"matches": results}