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
            "sentiment": prediction["sentiment"].title(),
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

# Simplified Gradio interface
demo = gr.Interface(
    fn=analyze,
    inputs=gr.Textbox(
        label="🔤 اردو متن درج کریں", 
        placeholder="مثال: یہ فلم بہت اچھی تھی...",
        lines=3
    ),
    outputs=gr.JSON(label="نتائج"),
    title="🇵🇰 اردو جذباتی تجزیہ",
    description="برٹ ماڈل کی مدد سے اردو متن کے جذبات کا تجزیہ کریں۔",
    examples=[
        ["میں آج بہت خوش ہوں"], 
        ["یہ فیلم بالکل اچھی نہیں تھی"],
        ["کھانا ذائقہ دار تھا مگر مہنگا تھا"]
    ],
    flagging_options=None,  # Replaces deprecated allow_flagging
    theme=gr.themes.Soft()
)

# Launch configuration optimized for Hugging Face Spaces
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=True
    )