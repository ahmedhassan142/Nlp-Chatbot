from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UrduSentimentPredictor:
    """Lightweight predictor for production use"""
    def __init__(self, model_path: str = None):
        try:
            # Option 1: Use a pre-trained model from Hugging Face Hub
            if model_path is None or model_path.startswith("bert-"):
                model_name = model_path or "bert-base-multilingual-cased"
                self.model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Option 2: Load local model files
            elif os.path.exists(model_path):
                self.model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Option 3: Handle invalid paths
            else:
                raise ValueError(f"Invalid model path: {model_path}. Must be either: "
                               "1) Hugging Face model ID, or "
                               "2) Path to local directory containing model files")
            
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