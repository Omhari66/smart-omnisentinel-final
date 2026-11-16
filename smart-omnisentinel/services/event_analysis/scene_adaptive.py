"""
services/event_analysis/scene_adaptive.py
-----------------------------------------
Day 9 Deliverable: Scene-Adaptive Dynamic Thresholding Engine.

Dynamically calibrates alert sensitivity based on environmental context:
  - sports: 0.85 (higher threshold suppresses energetic sports false positives)
  - office: 0.55 (strict sensitivity for quiet corporate settings)
  - street: 0.65 (balanced threshold for public streets)
  - playground: 0.80 (tolerates playful children movements)
  - indoor_generic: 0.60 (default baseline)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

from core.logger import get_logger

logger = get_logger(__name__)

DEFAULT_THRESHOLDS: Dict[str, float] = {
    "sports": 0.85,
    "office": 0.55,
    "street": 0.65,
    "playground": 0.80,
    "indoor_generic": 0.60,
    "_default": 0.60,
}

SCENE_CLASSIFICATION_PROMPT = """
Analyze the background and context of this surveillance frame.
Categorize the environment strictly into one of the following 5 labels:
- sports (basketball court, gym, football field, boxing ring)
- office (cubicles, conference room, corporate hallway)
- street (sidewalk, road, alleyway, parking lot)
- playground (park, children playing area, schoolyard)
- indoor_generic (store, mall, living room, lobby)

Output ONLY the category name in lowercase.
"""


class SceneAdaptiveManager:
    """
    Loads and applies per-scene dynamic thresholds from config/thresholds.json.
    """

    def __init__(self, config_path: Optional[str] = None):
        if config_path:
            self.config_path = Path(config_path)
        else:
            root = Path(__file__).parent.parent.parent
            # Check config/thresholds.json or configs/thresholds.json
            p1 = root / "config" / "thresholds.json"
            p2 = root / "configs" / "thresholds.json"
            self.config_path = p1 if p1.exists() else p2

        self._thresholds: Dict[str, float] = DEFAULT_THRESHOLDS.copy()
        self.load_thresholds()

    def load_thresholds(self) -> None:
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._thresholds.update(data)
                logger.info("scene_thresholds_loaded", path=str(self.config_path), count=len(self._thresholds))
            except Exception as e:
                logger.warning("scene_thresholds_load_failed", error=str(e), fallback="defaults")
        else:
            logger.info("scene_thresholds_using_defaults")

    def get_threshold(self, scene_type: Optional[str]) -> float:
        if not scene_type:
            return self._thresholds.get("_default", 0.60)
        return self._thresholds.get(scene_type.lower().strip(), self._thresholds.get("_default", 0.60))

    def should_trigger_alert(self, risk_score: float, scene_type: Optional[str]) -> bool:
        threshold = self.get_threshold(scene_type)
        return risk_score >= threshold


_scene_manager: Optional[SceneAdaptiveManager] = None


def get_scene_manager() -> SceneAdaptiveManager:
    global _scene_manager
    if _scene_manager is None:
        _scene_manager = SceneAdaptiveManager()
    return _scene_manager

