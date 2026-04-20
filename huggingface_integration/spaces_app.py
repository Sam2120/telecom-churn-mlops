"""Gradio app for Hugging Face Spaces deployment."""

import logging
import pickle
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model
model = None
pca_model = None


def load_model_from_hub(repo_id: str = "telecom-churn-model"):
    """Load model from Hugging Face Hub."""
    global model, pca_model
    
    try:
        # Download model files
        model_path = hf_hub_download(repo_id=repo_id, filename="churn_model.pkl")
        
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        # Try to download PCA model
        try:
            pca_path = hf_hub_download(repo_id=repo_id, filename="pca_model.pkl")
            with open(pca_path, "rb") as f:
                pca_model = pickle.load(f)
        except:
            pca_model = None
        
        logger.info("Model loaded successfully from Hugging Face Hub")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False


def predict_churn(
    recharge_m6: float,
    recharge_m7: float,
    recharge_m8: float,
    calls_m6: float,
    calls_m7: float,
    calls_m8: float,
    duration_m6: float,
    duration_m7: float,
    duration_m8: float,
    incoming_m6: float,
    incoming_m7: float,
    incoming_m8: float,
    outgoing_m6: float,
    outgoing_m7: float,
    outgoing_m8: float,
    data_m6: float,
    data_m7: float,
    data_m8: float
) -> tuple:
    """Make churn prediction."""
    if model is None:
        return "Model not loaded", "N/A", "N/A"
    
    # Create feature array
    features = np.array([[
        recharge_m6, recharge_m7, recharge_m8,
        calls_m6, calls_m7, calls_m8,
        duration_m6, duration_m7, duration_m8,
        incoming_m6, incoming_m7, incoming_m8,
        outgoing_m6, outgoing_m7, outgoing_m8,
        data_m6, data_m7, data_m8
    ]])
    
    # Apply PCA if available
    if pca_model is not None:
        features = pca_model.transform(features)
    
    # Predict
    probability = model.predict_proba(features)[0, 1]
    prediction = "Churn" if probability > 0.5 else "No Churn"
    
    # Risk level
    if probability >= 0.8:
        risk_level = "Very High"
    elif probability >= 0.6:
        risk_level = "High"
    elif probability >= 0.4:
        risk_level = "Medium"
    elif probability >= 0.2:
        risk_level = "Low"
    else:
        risk_level = "Very Low"
    
    # Confidence
    if abs(probability - 0.5) >= 0.4:
        confidence = "Very High"
    elif abs(probability - 0.5) >= 0.3:
        confidence = "High"
    elif abs(probability - 0.5) >= 0.2:
        confidence = "Medium"
    else:
        confidence = "Low"
    
    # Generate recommendations
    recommendations = []
    if probability >= 0.8:
        recommendations = [
            "Immediate intervention required",
            "Offer personalized retention package",
            "Schedule executive callback within 24 hours",
            "Waive upcoming monthly fees"
        ]
    elif probability >= 0.6:
        recommendations = [
            "Proactive outreach via preferred channel",
            "Offer targeted bundle promotions",
            "Provide loyalty rewards",
            "Resolve service issues proactively"
        ]
    elif probability >= 0.4:
        recommendations = [
            "Include in retention campaign",
            "Monitor usage patterns closely",
            "Send personalized offers",
            "Consider upgrade incentives"
        ]
    else:
        recommendations = [
            "Continue standard engagement",
            "Include in regular promotions",
            "Monitor for usage decline"
        ]
    
    result_text = f"""
## Prediction Results

**Churn Probability:** {probability:.1%}
**Prediction:** {prediction}
**Risk Level:** {risk_level}
**Confidence:** {confidence}

### Recommended Actions
"""
    for i, rec in enumerate(recommendations, 1):
        result_text += f"\n{i}. {rec}"
    
    return result_text, risk_level, f"{probability:.1%}"


# Create Gradio interface
def create_interface():
    """Create Gradio interface."""
    
    with gr.Blocks(title="Telecom Churn Predictor") as demo:
        gr.Markdown("""
        # Telecom Customer Churn Prediction
        
        This AI-powered tool predicts the likelihood of customer churn for high-value telecom customers.
        
        ### Instructions
        Enter customer usage data for months 6, 7, and 8 to get a churn prediction.
        
        - **Months 6-7:** Good phase (historical behavior)
        - **Month 8:** Action phase (current behavior for prediction)
        - **Prediction:** Churn risk for month 9
        """)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Recharge Amount (₹)")
                recharge_m6 = gr.Number(label="Month 6", value=500, minimum=0)
                recharge_m7 = gr.Number(label="Month 7", value=450, minimum=0)
                recharge_m8 = gr.Number(label="Month 8", value=300, minimum=0)
            
            with gr.Column():
                gr.Markdown("### Total Calls")
                calls_m6 = gr.Number(label="Month 6", value=150, minimum=0)
                calls_m7 = gr.Number(label="Month 7", value=140, minimum=0)
                calls_m8 = gr.Number(label="Month 8", value=80, minimum=0)
            
            with gr.Column():
                gr.Markdown("### Call Duration (seconds)")
                duration_m6 = gr.Number(label="Month 6", value=3000, minimum=0)
                duration_m7 = gr.Number(label="Month 7", value=2800, minimum=0)
                duration_m8 = gr.Number(label="Month 8", value=1500, minimum=0)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Incoming Calls")
                incoming_m6 = gr.Number(label="Month 6", value=50, minimum=0)
                incoming_m7 = gr.Number(label="Month 7", value=45, minimum=0)
                incoming_m8 = gr.Number(label="Month 8", value=20, minimum=0)
            
            with gr.Column():
                gr.Markdown("### Outgoing Calls")
                outgoing_m6 = gr.Number(label="Month 6", value=100, minimum=0)
                outgoing_m7 = gr.Number(label="Month 7", value=95, minimum=0)
                outgoing_m8 = gr.Number(label="Month 8", value=60, minimum=0)
            
            with gr.Column():
                gr.Markdown("### Data Usage (MB)")
                data_m6 = gr.Number(label="Month 6", value=2048, minimum=0)
                data_m7 = gr.Number(label="Month 7", value=1800, minimum=0)
                data_m8 = gr.Number(label="Month 8", value=512, minimum=0)
        
        with gr.Row():
            predict_btn = gr.Button("Predict Churn Risk", variant="primary")
        
        with gr.Row():
            with gr.Column():
                result_output = gr.Markdown(label="Results")
            with gr.Column():
                risk_indicator = gr.Label(label="Risk Level")
                prob_indicator = gr.Label(label="Churn Probability")
        
        # Example inputs
        gr.Examples(
            examples=[
                [500, 450, 300, 150, 140, 80, 3000, 2800, 1500, 50, 45, 20, 100, 95, 60, 2048, 1800, 512],
                [800, 850, 900, 200, 210, 220, 5000, 5200, 5500, 80, 85, 90, 120, 125, 130, 4096, 4096, 4096],
                [200, 150, 100, 50, 40, 20, 1000, 800, 400, 20, 15, 5, 30, 25, 15, 500, 300, 100],
            ],
            inputs=[
                recharge_m6, recharge_m7, recharge_m8,
                calls_m6, calls_m7, calls_m8,
                duration_m6, duration_m7, duration_m8,
                incoming_m6, incoming_m7, incoming_m8,
                outgoing_m6, outgoing_m7, outgoing_m8,
                data_m6, data_m7, data_m8
            ],
            label="Example Customers"
        )
        
        predict_btn.click(
            fn=predict_churn,
            inputs=[
                recharge_m6, recharge_m7, recharge_m8,
                calls_m6, calls_m7, calls_m8,
                duration_m6, duration_m7, duration_m8,
                incoming_m6, incoming_m7, incoming_m8,
                outgoing_m6, outgoing_m7, outgoing_m8,
                data_m6, data_m7, data_m8
            ],
            outputs=[result_output, risk_indicator, prob_indicator]
        )
        
        gr.Markdown("""
        ---
        ### About This Model
        
        - **Model Type:** Ensemble Machine Learning (Random Forest, Gradient Boosting, Logistic Regression)
        - **Training Data:** 4 months of telecom customer behavior
        - **High-Value Customers:** Top 70th percentile by recharge amount
        - **Accuracy:** AUC > 0.85 on test set
        - **MLOps Pipeline:** DVC, MLflow, CI/CD, Docker, FastAPI
        
        [View on GitHub](https://github.com/your-org/telecom-churn-mlops) | [Model Card](https://huggingface.co/your-model-repo)
        """)
    
    return demo


def main():
    """Launch the app."""
    # Try to load model from hub
    repo_id = "telecom-churn-model"  # Change to actual repo
    loaded = load_model_from_hub(repo_id)
    
    if not loaded:
        logger.warning("Running without loaded model - predictions will fail")
    
    # Create and launch interface
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
