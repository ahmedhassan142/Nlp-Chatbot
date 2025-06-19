import gradio as gr
from inference import UrduSentimentPredictor

predictor = UrduSentimentPredictor()

def format_response(prediction):
    if prediction.get("success"):
        return {
            "sentiment": prediction["sentiment"],
            "confidence": float(prediction["confidence"]),
            "status": "success"
        }
    else:
        print("❌ Backend prediction failed:", prediction["error"])
        return {
            "sentiment": None,
            "confidence": 0.0,
            "status": "error",
            "error": prediction["error"]
        }

def analyze(text):
    try:
        prediction = predictor.predict(text)
        return format_response(prediction)
    except Exception as e:
        return {"status": "error", "error": str(e)}

iface = gr.Interface(
    fn=analyze,
    inputs=gr.Textbox(label="Urdu Text", placeholder="Enter Urdu text here..."),
    outputs=gr.JSON(label="Analysis Result"),
    title="Urdu Sentiment Analyzer",
    examples=[["میں آج بہت خوش ہوں"], ["یہ فیلم بالکل اچھی نہیں تھی"]],
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch(debug=True, share=True,server_port=7861)
