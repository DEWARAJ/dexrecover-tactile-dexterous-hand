from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


class BehaviorCloningPolicy(nn.Module):
    def __init__(self, observation_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)


class NormalizedPolicy:
    def __init__(
        self,
        model: BehaviorCloningPolicy,
        obs_mean: np.ndarray,
        obs_std: np.ndarray,
        action_mean: np.ndarray,
        action_std: np.ndarray,
        action_low: np.ndarray,
        action_high: np.ndarray,
        feature_indices: np.ndarray,
    ):
        self.model = model.eval()
        self.obs_mean = np.asarray(obs_mean, dtype=np.float32)
        self.obs_std = np.asarray(obs_std, dtype=np.float32)
        self.action_mean = np.asarray(action_mean, dtype=np.float32)
        self.action_std = np.asarray(action_std, dtype=np.float32)
        self.action_low = np.asarray(action_low, dtype=np.float32)
        self.action_high = np.asarray(action_high, dtype=np.float32)
        self.feature_indices = np.asarray(feature_indices, dtype=np.int64)

    @torch.inference_mode()
    def __call__(self, observation: np.ndarray) -> np.ndarray:
        features = np.asarray(observation, dtype=np.float32)[self.feature_indices]
        normalized = (features - self.obs_mean) / self.obs_std
        output = self.model(torch.from_numpy(normalized).unsqueeze(0)).squeeze(0)
        command = output.cpu().numpy() * self.action_std + self.action_mean
        return np.clip(command, self.action_low, self.action_high).astype(np.float64)

    @classmethod
    def load(cls, checkpoint: str | Path) -> "NormalizedPolicy":
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = BehaviorCloningPolicy(
            int(payload["observation_size"]),
            int(payload["action_size"]),
            int(payload["hidden_size"]),
        )
        model.load_state_dict(payload["model_state"])
        return cls(
            model,
            payload["obs_mean"],
            payload["obs_std"],
            payload["action_mean"],
            payload["action_std"],
            payload["action_low"],
            payload["action_high"],
            payload["feature_indices"],
        )
