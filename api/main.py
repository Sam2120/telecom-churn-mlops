"""FastAPI application for Telecom Churn Prediction."""

import logging
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import MODELS_DIR, API_HOST, API_PORT, API_DEBUG, PROJECT_ROOT
from src.models import ModelExplainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models
model = None
pca_model = None
preprocessor = None
model_info = {}


class CustomerData(BaseModel):
    """Input data for single customer prediction."""
    customer_id: str = Field(..., description="Unique customer identifier")
    recharge_amount_m6: float = Field(..., ge=0, description="Recharge amount in month 6")
    recharge_amount_m7: float = Field(..., ge=0, description="Recharge amount in month 7")
    recharge_amount_m8: float = Field(..., ge=0, description="Recharge amount in month 8")
    total_calls_m6: float = Field(..., ge=0, description="Total calls in month 6")
    total_calls_m7: float = Field(..., ge=0, description="Total calls in month 7")
    total_calls_m8: float = Field(..., ge=0, description="Total calls in month 8")
    total_duration_m6: float = Field(..., ge=0, description="Total call duration in month 6")
    total_duration_m7: float = Field(..., ge=0, description="Total call duration in month 7")
    total_duration_m8: float = Field(..., ge=0, description="Total call duration in month 8")
    incoming_calls_m6: float = Field(..., ge=0, description="Incoming calls in month 6")
    incoming_calls_m7: float = Field(..., ge=0, description="Incoming calls in month 7")
    incoming_calls_m8: float = Field(..., ge=0, description="Incoming calls in month 8")
    outgoing_calls_m6: float = Field(..., ge=0, description="Outgoing calls in month 6")
    outgoing_calls_m7: float = Field(..., ge=0, description="Outgoing calls in month 7")
    outgoing_calls_m8: float = Field(..., ge=0, description="Outgoing calls in month 8")
    data_usage_m6: Optional[float] = Field(0, ge=0, description="Data usage in month 6 (MB)")
    data_usage_m7: Optional[float] = Field(0, ge=0, description="Data usage in month 7 (MB)")
    data_usage_m8: Optional[float] = Field(0, ge=0, description="Data usage in month 8 (MB)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "CUST001",
                "recharge_amount_m6": 500,
                "recharge_amount_m7": 450,
                "recharge_amount_m8": 300,
                "total_calls_m6": 150,
                "total_calls_m7": 140,
                "total_calls_m8": 80,
                "total_duration_m6": 3000,
                "total_duration_m7": 2800,
                "total_duration_m8": 1500,
                "incoming_calls_m6": 50,
                "incoming_calls_m7": 45,
                "incoming_calls_m8": 20,
                "outgoing_calls_m6": 100,
                "outgoing_calls_m7": 95,
                "outgoing_calls_m8": 60,
                "data_usage_m6": 2048,
                "data_usage_m7": 1800,
                "data_usage_m8": 512
            }
        }


class BatchPredictionRequest(BaseModel):
    """Input for batch predictions."""
    customers: List[CustomerData]


class PredictionResponse(BaseModel):
    """Response for churn prediction."""
    customer_id: str
    churn_probability: float = Field(..., ge=0, le=1)
    prediction: str
    risk_level: str
    confidence: str
    
    
class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    predictions: List[PredictionResponse]
    summary: Dict[str, Any]


class ModelInfoResponse(BaseModel):
    """Response with model information."""
    model_type: str
    model_path: str
    version: str
    metrics: Dict[str, float]
    feature_count: int
    last_updated: str


class ExplanationResponse(BaseModel):
    """Response with prediction explanation."""
    customer_id: str
    churn_probability: float
    prediction: str
    risk_level: str
    top_factors: List[Dict[str, Any]]
    recommendations: List[str]


def load_models():
    """Load trained models from disk."""
    global model, pca_model, preprocessor, model_info
    
    try:
        # Load main model
        model_path = MODELS_DIR / "churn_model.pkl"
        if model_path.exists():
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.warning(f"Model file not found: {model_path}")
        
        # Load PCA model
        pca_path = MODELS_DIR / "pca_model.pkl"
        if pca_path.exists():
            with open(pca_path, "rb") as f:
                pca_model = pickle.load(f)
            logger.info(f"Loaded PCA model from {pca_path}")
        
        # Load preprocessor
        preprocessor_path = MODELS_DIR / "preprocessor.pkl"
        if preprocessor_path.exists():
            with open(preprocessor_path, "rb") as f:
                preprocessor = pickle.load(f)
            logger.info(f"Loaded preprocessor from {preprocessor_path}")
        
        # Load model info
        import json
        info_path = MODELS_DIR / "model_info.json"
        if info_path.exists():
            with open(info_path, "r") as f:
                model_info = json.load(f)
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise


def get_feature_array(data: CustomerData) -> np.ndarray:
    """Convert CustomerData to feature array."""
    features = [
        data.recharge_amount_m6, data.recharge_amount_m7, data.recharge_amount_m8,
        data.total_calls_m6, data.total_calls_m7, data.total_calls_m8,
        data.total_duration_m6, data.total_duration_m7, data.total_duration_m8,
        data.incoming_calls_m6, data.incoming_calls_m7, data.incoming_calls_m8,
        data.outgoing_calls_m6, data.outgoing_calls_m7, data.outgoing_calls_m8,
        data.data_usage_m6 or 0, data.data_usage_m7 or 0, data.data_usage_m8 or 0
    ]
    return np.array(features).reshape(1, -1)


def get_risk_level(probability: float) -> str:
    """Determine risk level from probability."""
    if probability >= 0.8:
        return "Very High"
    elif probability >= 0.6:
        return "High"
    elif probability >= 0.4:
        return "Medium"
    elif probability >= 0.2:
        return "Low"
    else:
        return "Very Low"


def get_confidence(probability: float) -> str:
    """Determine confidence level."""
    if abs(probability - 0.5) >= 0.4:
        return "Very High"
    elif abs(probability - 0.5) >= 0.3:
        return "High"
    elif abs(probability - 0.5) >= 0.2:
        return "Medium"
    else:
        return "Low"


def generate_recommendations(probability: float, risk_level: str) -> List[str]:
    """Generate retention recommendations based on risk level."""
    recommendations = []
    
    if probability >= 0.8:
        recommendations.extend([
            "Immediate intervention required - assign dedicated retention specialist",
            "Offer personalized retention package with significant incentives",
            "Schedule executive callback within 24 hours",
            "Waive upcoming monthly fees or provide bill credits"
        ])
    elif probability >= 0.6:
        recommendations.extend([
            "Proactive outreach via preferred contact channel",
            "Offer targeted data/calling bundle promotions",
            "Provide loyalty rewards or bonus data allocation",
            "Analyze recent service issues and resolve proactively"
        ])
    elif probability >= 0.4:
        recommendations.extend([
            "Include in next retention campaign cycle",
            "Monitor usage patterns closely for next 30 days",
            "Send personalized offers based on usage history",
            "Consider upgrade incentives to increase engagement"
        ])
    else:
        recommendations.extend([
            "Continue standard engagement practices",
            "Include in regular promotional communications",
            "Monitor for any significant usage decline",
            "Maintain service quality standards"
        ])
    
    return recommendations


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("Loading models...")
    load_models()
    logger.info("API startup complete")
    
    yield
    
    # Shutdown
    logger.info("API shutdown")


# Create FastAPI app
app = FastAPI(
    title="Telecom Churn Prediction API",
    description="MLOps-driven API for predicting high-value customer churn in telecom industry",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Telecom Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_info": {
            "type": model_info.get("model_type", "unknown") if model_info else "unknown",
            "version": model_info.get("version", "unknown") if model_info else "unknown"
        }
    }


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfoResponse(
        model_type=model_info.get("model_type", "unknown"),
        model_path=str(MODELS_DIR / "churn_model.pkl"),
        version=model_info.get("version", "1.0.0"),
        metrics=model_info.get("metrics", {}),
        feature_count=model_info.get("feature_count", 0),
        last_updated=model_info.get("last_updated", "unknown")
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerData):
    """Make churn prediction for a single customer."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get features
        features = get_feature_array(customer)
        
        # Apply PCA if available
        if pca_model is not None:
            features = pca_model.transform(features)
        
        # Make prediction
        probability = float(model.predict_proba(features)[0, 1])
        prediction = "Churn" if probability > 0.5 else "No Churn"
        risk_level = get_risk_level(probability)
        confidence = get_confidence(probability)
        
        return PredictionResponse(
            customer_id=customer.customer_id,
            churn_probability=probability,
            prediction=prediction,
            risk_level=risk_level,
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Make churn predictions for multiple customers."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        predictions = []
        high_risk_count = 0
        
        for customer in request.customers:
            features = get_feature_array(customer)
            
            if pca_model is not None:
                features = pca_model.transform(features)
            
            probability = float(model.predict_proba(features)[0, 1])
            prediction = "Churn" if probability > 0.5 else "No Churn"
            risk_level = get_risk_level(probability)
            confidence = get_confidence(probability)
            
            if probability >= 0.6:
                high_risk_count += 1
            
            predictions.append(PredictionResponse(
                customer_id=customer.customer_id,
                churn_probability=probability,
                prediction=prediction,
                risk_level=risk_level,
                confidence=confidence
            ))
        
        # Calculate summary
        churn_count = sum(1 for p in predictions if p.prediction == "Churn")
        avg_probability = sum(p.churn_probability for p in predictions) / len(predictions)
        
        summary = {
            "total_customers": len(predictions),
            "predicted_churners": churn_count,
            "churn_rate": churn_count / len(predictions),
            "high_risk_customers": high_risk_count,
            "average_churn_probability": avg_probability
        }
        
        return BatchPredictionResponse(predictions=predictions, summary=summary)
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.post("/predict/file")
async def predict_file(file: UploadFile = File(...)):
    """Make predictions from uploaded CSV file."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read CSV
        contents = await file.read()
        from io import StringIO
        df = pd.read_csv(StringIO(contents.decode("utf-8")))
        
        # Validate required columns
        required_cols = ["customer_id"]
        if not all(col in df.columns for col in required_cols):
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {required_cols}"
            )
        
        # Make predictions
        predictions = []
        for _, row in df.iterrows():
            # Convert row to features (simplified - assumes correct column names)
            features = row.drop("customer_id").values.reshape(1, -1)
            
            if pca_model is not None:
                features = pca_model.transform(features)
            
            probability = float(model.predict_proba(features)[0, 1])
            prediction = "Churn" if probability > 0.5 else "No Churn"
            risk_level = get_risk_level(probability)
            
            predictions.append({
                "customer_id": row["customer_id"],
                "churn_probability": probability,
                "prediction": prediction,
                "risk_level": risk_level
            })
        
        return {
            "filename": file.filename,
            "predictions": predictions,
            "count": len(predictions)
        }
        
    except Exception as e:
        logger.error(f"File prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"File prediction failed: {str(e)}")


@app.post("/explain", response_model=ExplanationResponse)
async def explain_prediction(customer: CustomerData):
    """Explain churn prediction for a customer."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get prediction
        features = get_feature_array(customer)
        
        if pca_model is not None:
            features = pca_model.transform(features)
        
        probability = float(model.predict_proba(features)[0, 1])
        prediction = "Churn" if probability > 0.5 else "No Churn"
        risk_level = get_risk_level(probability)
        
        # Generate explanation using model-specific methods
        top_factors = []
        
        # Get feature names
        feature_names = [
            "recharge_amount_m6", "recharge_amount_m7", "recharge_amount_m8",
            "total_calls_m6", "total_calls_m7", "total_calls_m8",
            "total_duration_m6", "total_duration_m7", "total_duration_m8",
            "incoming_calls_m6", "incoming_calls_m7", "incoming_calls_m8",
            "outgoing_calls_m6", "outgoing_calls_m7", "outgoing_calls_m8",
            "data_usage_m6", "data_usage_m7", "data_usage_m8"
        ]
        
        # If using interpretable model, get feature importance
        if hasattr(model.named_steps.get("classifier", model), "coef_"):
            coef = model.named_steps["classifier"].coef_[0]
            contributions = features[0] * coef
            
            # Get top contributing features
            contrib_list = list(zip(feature_names[:len(contributions)], contributions))
            contrib_list.sort(key=lambda x: abs(x[1]), reverse=True)
            
            top_factors = [
                {
                    "feature": f,
                    "contribution": float(c),
                    "impact": "Positive" if c > 0 else "Negative"
                }
                for f, c in contrib_list[:5]
            ]
        else:
            # Fallback: use feature values as proxy
            values = features[0]
            value_list = list(zip(feature_names[:len(values)], values))
            value_list.sort(key=lambda x: abs(x[1]), reverse=True)
            
            top_factors = [
                {
                    "feature": f,
                    "value": float(v),
                    "impact": "High" if abs(v) > np.mean(np.abs(values)) else "Low"
                }
                for f, v in value_list[:5]
            ]
        
        # Generate recommendations
        recommendations = generate_recommendations(probability, risk_level)
        
        return ExplanationResponse(
            customer_id=customer.customer_id,
            churn_probability=probability,
            prediction=prediction,
            risk_level=risk_level,
            top_factors=top_factors,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
