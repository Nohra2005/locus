# LOCUS: main.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "online", "service": "Locus Gateway"}

@app.get("/health")
def health_check():
    return {"health": "ok"}