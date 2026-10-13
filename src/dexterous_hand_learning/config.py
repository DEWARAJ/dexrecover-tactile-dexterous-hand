from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 7
    control_dt: float = 0.02
    episode_steps: int = 220
    demonstrations: int = 80
    evaluation_episodes: int = 50
    train_epochs: int = 120
    batch_size: int = 256
    learning_rate: float = 1e-3
    hidden_size: int = 128
    success_radius_m: float = 0.055
    max_action_age_s: float = 0.10

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

