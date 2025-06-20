from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UrduSentimentPredictor:
    """Lightweight predictor for production use"""
    def __init__(self, model_path: str = "Ahmedhassan54/Nlp-chatbot/trained_model"):  # Changed default path
        try:
            # Now only needs Hub loading (Option 1)
            self.model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            logger.info(f"Model loaded successfully from Hugging Face Hub: {model_path}")
            
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
                "sentiment": str(label),
                "confidence": float(confidence),
                "success": True
            }
        except Exception as e:
            return {
                "error": str(e),
                "success": False,
                "sentiment": None,
                "confidence": 0.0
            }

# Usage example
predictor = UrduSentimentPredictor()  # Will automatically load from Ahmedhassan54/Nlp-chatbot
result = predictor.predict("آپ کیسے ہیں؟")
print(result)