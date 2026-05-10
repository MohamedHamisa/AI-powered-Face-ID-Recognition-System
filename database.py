"""
database.py — JSON-backed face embedding database.
Each record: { "name": str, "embedding": List[float] }
"""

import json
import os
import numpy as np
from typing import List, Dict, Optional


class FaceDatabase:
    def __init__(self, path: str = "embeddings.json"):
        self.path = path
        self.data: Dict[str, List[float]] = {}
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────
    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    self.data = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.data = {}

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    # ── CRUD ──────────────────────────────────────────────────────────────────
    def add_person(self, name: str, embedding: List[float]):
        """Add or overwrite a person's embedding."""
        self.data[name] = embedding
        self._save()

    def remove_person(self, name: str):
        if name in self.data:
            del self.data[name]
            self._save()

    def get_embedding(self, name: str) -> Optional[np.ndarray]:
        if name in self.data:
            return np.array(self.data[name])
        return None

    def get_all_names(self) -> List[str]:
        return list(self.data.keys())

    def get_all_embeddings(self) -> List[tuple]:
        """Returns list of (name, np.ndarray) tuples."""
        return [(name, np.array(emb)) for name, emb in self.data.items()]

    def clear(self):
        self.data = {}
        self._save()

    def __len__(self):
        return len(self.data)
