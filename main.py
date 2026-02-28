from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import sys
import os

# Ensure parent directory is in path to import data_loader and metrics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from data_loader import fetch_data
    from metrics import calculate_metrics, perform_cross_sectional_regression
except ImportError as e:
    # Handle the case where the script is run directly from the parent or backend directory indifferently
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from ..data_loader import fetch_data
        from ..metrics import calculate_metrics, perform_cross_sectional_regression
    except ImportError:
        pass

app = FastAPI(title="Risk-Return API")

# Setup CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    tickers: str
    benchmark: str = "^GSPC"
    years: int = 5
    risk_free_rate: float = 0.04

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Risk-Return API is running"}

@app.post("/api/analyze")
def analyze_funds(req: AnalysisRequest):
    try:
        # 1. Fetch Data
        returns_df = fetch_data(req.tickers, benchmark_ticker=req.benchmark, years=req.years)
        
        if returns_df.empty:
            raise HTTPException(status_code=400, detail="No data fetched. Please check the tickers.")
            
        # 2. Calculate Metrics
        metrics_df = calculate_metrics(returns_df, benchmark_ticker=req.benchmark, risk_free_rate=req.risk_free_rate)
        
        if metrics_df.empty:
            raise HTTPException(status_code=500, detail="Error calculating metrics.")
            
        # 3. Perform Regression
        sml_model = perform_cross_sectional_regression(metrics_df)
        
        # 4. Format Output for JSON Response
        # Convert metrics dataframe to list of dictionaries
        metrics_list = metrics_df.replace({np.nan: None}).to_dict(orient="records")
        
        # Extract sml parameters
        sml_data = {
            "slope": float(sml_model["slope"]),
            "intercept": float(sml_model["intercept"]),
            "r_squared": float(sml_model["r_squared"])
        }
        
        # Generate trendline data points
        min_beta = float(metrics_df["Beta"].min()) * 0.9
        max_beta = float(metrics_df["Beta"].max()) * 1.1
        
        trendline = [
            {"x": min_beta, "y": (sml_data["slope"] * min_beta) + sml_data["intercept"]},
            {"x": max_beta, "y": (sml_data["slope"] * max_beta) + sml_data["intercept"]}
        ]
        
        return {
            "status": "success",
            "metrics": metrics_list,
            "sml": sml_data,
            "trendline": trendline
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # When running directly `python backend/main.py`
    uvicorn.run(app, host="0.0.0.0", port=8000)
