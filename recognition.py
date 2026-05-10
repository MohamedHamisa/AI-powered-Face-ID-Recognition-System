"""
recognition.py — Ties together the model and database:
  - extracts embeddings from a frame
  - compares against the database (cosine similarity)
  - annotates the image with bounding boxes and labels
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple
from model import get_face_detector
from database import FaceDatabase


# ── Visual constants ──────────────────────────────────────────────────────────
COLOR_RECOGNIZED = (0, 220, 100)   # green
COLOR_UNKNOWN    = (0, 100, 255)   # orange-red (BGR)
FONT             = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE       = 0.7
THICKNESS        = 2


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class FaceRecognizer:
    def __init__(self, db: FaceDatabase, threshold: float = 0.55):
        self.db = db
        self.threshold = threshold          # cosine similarity cutoff
        self._model = get_face_detector()

    # ── Core inference ────────────────────────────────────────────────────────
    def extract_embeddings(self, img_bgr: np.ndarray) -> List[np.ndarray]:
        """Return list of raw embeddings for all faces found in the image."""
        detections = self._model.detect_and_embed(img_bgr)
        return [emb for emb, _ in detections]

    def recognize_image(
        self, img_bgr: np.ndarray
    ) -> Tuple[List[Dict], np.ndarray]:
        """
        Detect + recognize all faces in img_bgr.
        Returns:
            results   : list of dicts {name, recognized, confidence, bbox}
            annotated : annotated BGR image
        """
        annotated = img_bgr.copy()
        detections = self._model.detect_and_embed(img_bgr)

        results = []
        for embedding, bbox in detections:
            name, confidence = self._match(embedding)
            recognized = confidence >= self.threshold

            result = {
                "name": name if recognized else "Unknown",
                "recognized": recognized,
                "confidence": confidence,
                "bbox": bbox,
            }
            results.append(result)
            self._draw(annotated, bbox, result["name"], confidence, recognized)

        return results, annotated

    # ── Matching ──────────────────────────────────────────────────────────────
    def _match(self, embedding: np.ndarray) -> Tuple[str, float]:
        """Find best-matching person via cosine similarity."""
        all_persons = self.db.get_all_embeddings()
        if not all_persons:
            return "Unknown", 0.0

        best_name, best_score = "Unknown", 0.0
        for name, stored_emb in all_persons:
            score = cosine_similarity(embedding, stored_emb)
            if score > best_score:
                best_score = score
                best_name = name

        return best_name, best_score

    # ── Drawing ───────────────────────────────────────────────────────────────
    def _draw(
        self,
        img: np.ndarray,
        bbox: List[int],
        label: str,
        confidence: float,
        recognized: bool,
    ):
        x1, y1, x2, y2 = bbox
        color = COLOR_RECOGNIZED if recognized else COLOR_UNKNOWN

        # Bounding box with rounded corners effect (double rect)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, THICKNESS)
        cv2.rectangle(img, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), color, 1)

        # Label background
        display = f"{label}" if not recognized else f"{label}  {confidence:.0%}"
        (tw, th), baseline = cv2.getTextSize(display, FONT, FONT_SCALE, THICKNESS)
        label_y1 = max(y1 - th - 10, 0)
        cv2.rectangle(img, (x1, label_y1), (x1 + tw + 10, y1), color, -1)

        # Label text (dark on coloured bg)
        cv2.putText(
            img, display,
            (x1 + 5, y1 - 5),
            FONT, FONT_SCALE,
            (10, 10, 10), THICKNESS, cv2.LINE_AA,
        )

        # Confidence bar (only when recognized)
        if recognized:
            bar_w = x2 - x1
            filled = int(bar_w * confidence)
            cv2.rectangle(img, (x1, y2 + 2), (x2, y2 + 6), (40, 40, 40), -1)
            cv2.rectangle(img, (x1, y2 + 2), (x1 + filled, y2 + 6), color, -1)
