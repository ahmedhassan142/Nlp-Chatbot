import gradio as gr
from inference import UrduSentimentPredictor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize predictor - automatically falls back to bert-base-multilingual-cased
predictor = UrduSentimentPredictor()

def format_response(prediction):
    """Standardize API response format"""
    if prediction.get("success"):
        return {
            "sentiment": prediction["sentiment"].title(),  # Capitalize (Positive/Negative)
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

# Gradio interface with improved UI
demo = gr.Interface(
    fn=analyze,
    inputs=gr.Textbox(
        label="🔤 اردو متن درج کریں", 
        placeholder="مثال: یہ فلم بہت اچھی تھی...",
        lines=3
    ),
    outputs=[
        gr.Label(label="جذبات"),
        gr.Label(label="اعتماد"),
        gr.Textbox(label="تفصیلات", visible=False)  # Hidden for error cases
    ],
    title="🇵🇰 اردو جذباتی تجزیہ",
    description="برٹ ماڈل کی مدد سے اردو متن کے جذبات کا تجزیہ کریں۔",
    examples=[
        ["میں آج بہت خوش ہوں"], 
        ["یہ فیلم بالکل اچھی نہیں تھی"],
        ["کھانا ذائقہ دار تھا مگر مہنگا تھا"]
    ],
    allow_flagging="never",
    theme=gr.themes.Soft()
)

def process_output(prediction):
    """Transform output for Gradio components"""
    if prediction["success"]:
        return (
            prediction["sentiment"],  # For Label 1
            prediction["confidence"], # For Label 2
            ""                        # Hide details box
        )
    return (
        "Error", 
        "0%", 
        f"❌ {prediction['error']}"  # Show error in details box
    )

# Reconfigure to use custom output processing
demo = gr.Interface(
    fn=analyze,
    inputs=demo.inputs,
    outputs=demo.outputs,
    examples=demo.examples,
    title=demo.title,
    description=demo.description,
    theme=demo.theme,
    allow_flagging="never",
    interpretation=None,
    live=False
)

# Launch configuration optimized for Hugging Face Spaces
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        debug=True,
        share=True
    )