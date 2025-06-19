from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
import pandas as pd
from urduhack.tokenization import word_tokenizer
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import warnings
import json
import numpy as np

# Suppress TensorFlow Addons deprecation warning
warnings.filterwarnings("ignore", message="TensorFlow Addons")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    model_name: str = "bert-base-multilingual-cased"
    num_labels: int = 2

class UrduSentimentTrainer:
    def __init__(self, config=ModelConfig()):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._initialize_tokenizer()

    def _initialize_tokenizer(self):
        """Initialize tokenizer with fallbacks"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                cache_dir='./models'
            )
        except Exception as e:
            logger.warning(f"Tokenizer init failed: {e}")
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

    def preprocess_text(self, text: str) -> str:
        """Preprocess with urduhack"""
        try:
            return " ".join(word_tokenizer(text)) if isinstance(text, str) else ""
        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            return text

    def _prepare_dataset(self, data: dict) -> tf.data.Dataset:
        """Convert raw data to TF Dataset"""
        df = pd.DataFrame(data)
        df["text"] = df["text"].apply(self.preprocess_text)
        
        encodings = self.tokenizer(
            df["text"].tolist(),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="tf"
        )
        
        return tf.data.Dataset.from_tensor_slices((
            dict(encodings),
            tf.constant(df["label"].tolist())
        )).batch(16)

    def train(self, data: dict, epochs: int = 3):
        """Train and save model with validation"""
        if not data or len(data.get("text", [])) == 0:
            raise ValueError("Training data cannot be empty")
            
        logger.info(f"Starting training with {len(data['text'])} samples")
        
        try:
            # Verify data
            sample_text = data["text"][0]
            sample_label = data["label"][0]
            logger.info(f"Sample data - Text: {sample_text}, Label: {sample_label}")
            
            # Initialize model
            self.model = TFAutoModelForSequenceClassification.from_pretrained(
                self.config.model_name,
                num_labels=self.config.num_labels
            )
            
            # Print model summary
            self.model.summary()
            
            # Compile model
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(3e-5),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=["accuracy"]
            )
            
            # Prepare dataset
            train_dataset = self._prepare_dataset(data)
            
            # Verify dataset
            sample_batch = next(iter(train_dataset))
            logger.info(f"Sample batch - Input: {sample_batch[0]}, Label: {sample_batch[1]}")
            
            # Train
            history = self.model.fit(
                train_dataset,
                epochs=epochs,
                verbose=1
            )
            
            logger.info("Training completed successfully")
            self.save_model()
            return history
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def save_model(self, output_dir: str = "saved_model"):
        """Save trained model"""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")


def save_training_data(data: dict, path: str = "data/train.json"):
    """Save training data"""
    Path("data").mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # Example usage
    trainer = UrduSentimentTrainer()
    
    # Sample data (replace with your actual data)
    sample_data = {
        "text": ["میں بہت خوش ہوں", "مجھے یہ پسند نہیں آیا"],
        "label": [1, 0]  # 1=positive, 0=negative
    }
    
    # Train and save model
    trainer.train(sample_data, epochs=3)