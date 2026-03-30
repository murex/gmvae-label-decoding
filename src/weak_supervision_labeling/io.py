# scr/weak_supervision_labeling/io.py

from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json

def make_run_root(base_dir: str = "runs") -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root = Path(base_dir) / ts
    root.mkdir(parents=True, exist_ok=True)
    return root

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ckpt_dir_for(method_tag: str, base_dir: str = "checkpoints") -> Path:
    """
    Dossier stable où sauver/charger un modèle,
    indépendant des runs horodatés.
    """
    p = Path(base_dir) / method_tag
    p.mkdir(parents=True, exist_ok=True)
    return p