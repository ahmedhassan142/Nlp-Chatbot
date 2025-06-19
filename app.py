import gradio as gr
from inference import UrduSentimentPredictor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize predictor
predictor = UrduSentimentPredictor()

def format_response(prediction):
    """Standardize API response format"""
    if prediction.get("success"):
        return {
            "sentiment": prediction["sentiment"].title(),  # "Positive"/"Negative"
            "confidence": f"{float(prediction['confidence']) * 100:.1f}%",
            "success": True
        }
    return {
        "error": prediction.get("error", "Unknown error"),
        "success": False
    }

def analyze(text):
    """Handle prediction with comprehensive error handling"""
    if not text.strip():
        return {"error": "Input text cannot be empty", "success": False}
    
    try:
        prediction = predictor.predict(text)
        return format_response(prediction)
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return {"error": str(e), "success": False}

# Gradio interface with English UI but Urdu examples
demo = gr.Interface(
    fn=analyze,
    inputs=gr.Textbox(
        label="Enter Urdu Text",  # English label
        placeholder="Example: یہ فلم بہت اچھی تھی...",  # Urdu placeholder
        lines=3
    ),
    outputs=gr.JSON(label="Analysis Results"),  # English label
    title="Urdu Sentiment Analyzer",  # English title
    description="Analyze sentiment of Urdu text using BERT model.",  # English description
    examples=[
        ["میں آج بہت خوش ہوں"],  # Urdu example 1
        ["یہ فیلم بالکل اچھی نہیں تھی"],  # Urdu example 2
        ["کھانا ذائقہ دار تھا مگر مہنگا تھا"]  # Urdu example 3
    ],
    flagging_options=None,
    theme=gr.themes.Soft()
)

# Launch configuration
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=True
    )