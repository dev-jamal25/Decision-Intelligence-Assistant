"""ML priority prediction service."""

import logging
import joblib
import json
import re
from pathlib import Path
from functools import lru_cache
import numpy as np

logger = logging.getLogger(__name__)


class PriorityPrediction:
    """Result from priority prediction."""

    def __init__(self, priority: str, confidence: float, model_name: str = "combined_logreg_v1"):
        self.priority = priority
        self.confidence = confidence
        self.model_name = model_name

    def to_dict(self) -> dict:
        return {
            "priority": self.priority,
            "confidence": float(self.confidence),
            "model": self.model_name,
        }


class PriorityService:
    """ML-based priority prediction for support tickets."""

    ARTIFACT_DIR = Path(__file__).parent.parent.parent.parent / "artifacts" / "models"

    def __init__(self):
        """Load ML artifacts."""
        self.model_name = "combined_logreg_v1"

        # Load model and metadata
        model_path = self.ARTIFACT_DIR / f"{self.model_name}.pkl"
        metadata_path = self.ARTIFACT_DIR / f"{self.model_name}_metadata.json"

        self.model = joblib.load(model_path)
        with open(metadata_path) as f:
            self.metadata = json.load(f)

        self.vectorizer = joblib.load(self.ARTIFACT_DIR / "tfidf_vectorizer_v1.pkl")
        self.scaler = joblib.load(self.ARTIFACT_DIR / "standard_scaler_v1.pkl")

        self.target_classes = self.metadata["target_classes"]  # ["normal", "urgent"]
        self.positive_class = self.metadata["positive_class"]  # "urgent"

        logger.info(f"Loaded priority model: {self.model_name}")
        logger.info(f"Target classes: {self.target_classes}")

    def extract_engineered_features(self, text: str) -> dict:
        """Extract engineered features from text."""
        return {
            "char_count": len(text),
            "word_count": len(text.split()),
            "exclamation_count": text.count("!"),
            "question_mark_count": text.count("?"),
            "caps_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
        }

    def predict(self, query: str) -> PriorityPrediction:
        """
        Predict priority for a query.

        Args:
            query: Support query text

        Returns:
            PriorityPrediction with priority and confidence
        """
        try:
            # Extract engineered features
            eng_features = self.extract_engineered_features(query)
            eng_feature_values = np.array(
                [
                    [
                        eng_features["char_count"],
                        eng_features["word_count"],
                        eng_features["exclamation_count"],
                        eng_features["question_mark_count"],
                        eng_features["caps_ratio"],
                    ]
                ]
            )

            # Scale engineered features
            scaled_eng_features = self.scaler.transform(eng_feature_values)

            # Vectorize text (TF-IDF)
            text_features = self.vectorizer.transform([query]).toarray()

            # Combine: TF-IDF features + scaled engineered features
            combined_features = np.hstack([text_features, scaled_eng_features])

            # Predict
            prediction = self.model.predict(combined_features)[0]
            probabilities = self.model.predict_proba(combined_features)[0]

            # Map to class name and get confidence
            priority_class = self.target_classes[prediction]
            confidence = probabilities[prediction]

            logger.info(f"Predicted priority: {priority_class} (confidence: {confidence:.4f})")
            return PriorityPrediction(
                priority=priority_class,
                confidence=confidence,
                model_name=self.model_name,
            )

        except Exception as e:
            logger.error(f"Priority prediction failed: {e}", exc_info=True)
            # Return neutral prediction on error
            return PriorityPrediction(priority="normal", confidence=0.5, model_name=self.model_name)


@lru_cache(maxsize=1)
def get_priority_service() -> PriorityService:
    """Get or create cached priority service."""
    return PriorityService()
