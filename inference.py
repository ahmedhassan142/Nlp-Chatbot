from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UrduSentimentPredictor:
    """Lightweight predictor for production use (no urduhack dependency)"""
    def __init__(self, model_path: str = "saved_model"):
        try:
            self.model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise

    def predict(self, text: str) -> dict:
        """Make prediction with JSON-safe outputs"""
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="tf",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            outputs = self.model(**inputs)

            probs = tf.nn.softmax(outputs.logits, axis=1)
            confidence = float(np.max(probs))
            label = "positive" if np.argmax(probs) == 1 else "negative"
            return {
            "sentiment": str(label),  # Explicit string conversion
            "confidence": float(confidence),  # Force Python float
            "success": True
        }
        except Exception as e:
             return {
            "error": str(e),  # Convert exception to string
            "success": False,
            "sentiment": None,  # Ensure consistent keys
            "confidence": 0.0
        }