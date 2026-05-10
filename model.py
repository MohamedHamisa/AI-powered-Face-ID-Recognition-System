"""
model.py — Deep learning face embedding model using FaceNet (InceptionResnetV1)
via the facenet-pytorch library.

FIXED VERSION:
- Removes ambiguous numpy array comparison error
- Faster processing (MTCNN called once instead of per face)
- Cleaner indexing using enumerate
- Better safety checks
"""

import torch
import numpy as np
import cv2

from typing import List, Tuple

# Try importing facenet-pytorch
try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
    FACENET_AVAILABLE = True
except ImportError:
    FACENET_AVAILABLE = False


# Device selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FaceEmbeddingModel:
    """
    Face recognition embedding model.
    Uses:
    - FaceNet (preferred)
    - OpenCV fallback if FaceNet unavailable
    """

    EMBEDDING_DIM = 512

    def __init__(self):

        if FACENET_AVAILABLE:
            self._init_facenet()
        else:
            self._init_fallback()

    # ============================================================
    # FACENET INITIALIZATION
    # ============================================================

    def _init_facenet(self):

        self.mode = "facenet"

        self.mtcnn = MTCNN(
            image_size=160,
            margin=20,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            keep_all=True,
            device=DEVICE
        )

        self.resnet = (
            InceptionResnetV1(pretrained="vggface2")
            .eval()
            .to(DEVICE)
        )

        print(f"[INFO] FaceNet initialized on {DEVICE}")

    # ============================================================
    # FACENET EMBEDDING
    # ============================================================

    def _embed_facenet(self, face_tensor: torch.Tensor) -> np.ndarray:
        """
        Generate normalized 512D embedding.
        """

        with torch.no_grad():

            embedding = self.resnet(
                face_tensor.unsqueeze(0).to(DEVICE)
            )

        embedding = torch.nn.functional.normalize(
            embedding,
            p=2,
            dim=1
        )

        return embedding.cpu().numpy()[0]

    # ============================================================
    # MAIN DETECTION FUNCTION
    # ============================================================

    def detect_and_embed(
        self,
        img_bgr: np.ndarray
    ) -> List[Tuple[np.ndarray, List[int]]]:

        """
        Returns:
            [
                (embedding, [x1, y1, x2, y2]),
                ...
            ]
        """

        if img_bgr is None:
            return []

        if self.mode == "facenet":

            img_rgb = cv2.cvtColor(
                img_bgr,
                cv2.COLOR_BGR2RGB
            )

            return self._detect_embed_facenet(img_rgb)

        else:
            return self._detect_embed_fallback(img_bgr)

    # ============================================================
    # FACENET DETECTION
    # ============================================================

    def _detect_embed_facenet(self, img_rgb: np.ndarray):

        from PIL import Image as PILImage

        pil_img = PILImage.fromarray(img_rgb)

        # Detect faces
        boxes, _ = self.mtcnn.detect(pil_img)

        # No faces
        if boxes is None or len(boxes) == 0:
            return []

        # Extract aligned face tensors ONCE
        face_tensors = self.mtcnn(pil_img)

        if face_tensors is None:
            return []

        results = []

        # Multiple faces
        if face_tensors.dim() == 4:

            for idx, box in enumerate(boxes):

                x1, y1, x2, y2 = [
                    max(0, int(v))
                    for v in box
                ]

                # Safety check
                if idx >= face_tensors.shape[0]:
                    continue

                face_tensor = face_tensors[idx]

                embedding = self._embed_facenet(face_tensor)

                results.append(
                    (
                        embedding,
                        [x1, y1, x2, y2]
                    )
                )

        # Single face
        else:

            box = boxes[0]

            x1, y1, x2, y2 = [
                max(0, int(v))
                for v in box
            ]

            embedding = self._embed_facenet(face_tensors)

            results.append(
                (
                    embedding,
                    [x1, y1, x2, y2]
                )
            )

        return results

    # ============================================================
    # FALLBACK INITIALIZATION
    # ============================================================

    def _init_fallback(self):

        self.mode = "fallback"

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
        )

        print("[INFO] Using OpenCV fallback detector")

    # ============================================================
    # FALLBACK DETECTOR
    # ============================================================

    def _detect_embed_fallback(self, img_bgr: np.ndarray):

        gray = cv2.cvtColor(
            img_bgr,
            cv2.COLOR_BGR2GRAY
        )

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        results = []

        for (x, y, w, h) in faces:

            face_gray = gray[y:y+h, x:x+w]

            face_resized = cv2.resize(
                face_gray,
                (128, 128)
            )

            # HOG descriptor
            hog = cv2.HOGDescriptor(
                (128, 128),
                (16, 16),
                (8, 8),
                (8, 8),
                9
            )

            descriptor = hog.compute(
                face_resized
            ).flatten()

            # Normalize to fixed size
            embedding = np.zeros(
                self.EMBEDDING_DIM,
                dtype=np.float32
            )

            n = min(
                len(descriptor),
                self.EMBEDDING_DIM
            )

            embedding[:n] = descriptor[:n]

            norm = np.linalg.norm(embedding)

            if norm > 0:
                embedding /= norm

            results.append(
                (
                    embedding,
                    [x, y, x + w, y + h]
                )
            )

        return results


# ============================================================
# SINGLETON ACCESS
# ============================================================

def get_face_detector():

    if not hasattr(get_face_detector, "_instance"):

        get_face_detector._instance = FaceEmbeddingModel()

    return get_face_detector._instance