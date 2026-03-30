# src/weak_supervision_labeling/base.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
from pathlib import Path

@dataclass
class FitOutput:
    embedding: Optional[np.ndarray] = None     # (N, d)
    labels: Optional[np.ndarray] = None        # (N,)
    proba: Optional[np.ndarray] = None         # (N, K) si soft
    extra: Optional[Dict[str, Any]] = None

class Method:
    name: str = "base"

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Method":
        return self

    def embed(self, X: np.ndarray) -> np.ndarray:
        """Retourne une représentation (N, d)."""
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Retourne des labels hard (N,)."""
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Retourne des probas (N, K) si disponible."""
        return None

    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> FitOutput:
        self.fit(X, y)
        emb = None
        try:
            emb = self.embed(X)
        except Exception:
            pass
        labels = self.predict(X)
        proba = self.predict_proba(X)
        return FitOutput(embedding=emb, labels=labels, proba=proba, extra={})
    
    def can_save(self) -> bool:
        """Retourne True si la méthode supporte save/load."""
        return False

    def save(self, ckpt_dir: Path) -> None:
        """Sauvegarde l'état entraîné dans ckpt_dir. (No-op par défaut)"""
        return None

    def load(self, ckpt_dir: Path) -> bool:
        """
        Charge un modèle depuis ckpt_dir.
        Retourne True si chargé, False sinon.
        """
        return False
    
    def can_generate_by_cluster(self) -> bool:
        return False

    def generate_by_cluster(
        self,
        cluster_id: int,
        n: int,
        device: str = "cuda",
    ) -> np.ndarray:
        """
        Retourne n échantillons x générés pour le cluster donné.
        Shape: (n, x_dim)
        """
        raise NotImplementedError