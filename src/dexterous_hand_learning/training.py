from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import ExperimentConfig
from .policy import BehaviorCloningPolicy


def train_policy(dataset_path: str | Path, checkpoint_path: str | Path, config: ExperimentConfig) -> dict:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    data = np.load(dataset_path)
    observations = data["observations"].astype(np.float32)
    actions = data["actions"].astype(np.float32)
    episode_ids = data["episode_ids"]
    split_episode = int(config.demonstrations * 0.8)
    train_mask = episode_ids < split_episode
    val_mask = ~train_mask

    # Keep the control state, cube pose, target and task phase. Raw body
    # velocities are noisy and caused rollout covariate shift despite low
    # held-out sample error.
    feature_indices = np.asarray([*range(7), 14, 15, 16, 26, 27, 28], dtype=np.int64)
    features = observations[:, feature_indices]
    obs_mean = features[train_mask].mean(axis=0)
    obs_std = features[train_mask].std(axis=0)
    obs_std = np.maximum(obs_std, 1e-5)
    obs_norm = (features - obs_mean) / obs_std
    action_mean = actions[train_mask].mean(axis=0)
    action_std = np.maximum(actions[train_mask].std(axis=0), 1e-5)
    action_norm = (actions - action_mean) / action_std

    train_ds = TensorDataset(torch.from_numpy(obs_norm[train_mask]), torch.from_numpy(action_norm[train_mask]))
    val_x = torch.from_numpy(obs_norm[val_mask])
    val_y = torch.from_numpy(action_norm[val_mask])
    loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, generator=torch.Generator().manual_seed(config.seed))

    model = BehaviorCloningPolicy(features.shape[1], actions.shape[1], config.hidden_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    history = {"train_loss": [], "validation_loss": []}
    best_state = None
    best_validation = float("inf")
    start = time.perf_counter()

    for _ in range(config.train_epochs):
        model.train()
        batch_losses = []
        for x_batch, y_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            validation = float(loss_fn(model(val_x), val_y))
        train_loss = float(np.mean(batch_losses))
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation)
        if validation < best_validation:
            best_validation = validation
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "obs_mean": obs_mean,
            "obs_std": obs_std,
            "action_mean": action_mean,
            "action_std": action_std,
            "action_low": actions.min(axis=0),
            "action_high": actions.max(axis=0),
            "feature_indices": feature_indices,
            "observation_size": features.shape[1],
            "action_size": actions.shape[1],
            "hidden_size": config.hidden_size,
        },
        checkpoint_path,
    )
    report = {
        "train_samples": int(train_mask.sum()),
        "validation_samples": int(val_mask.sum()),
        "epochs": config.train_epochs,
        "best_validation_mse": best_validation,
        "final_train_mse": history["train_loss"][-1],
        "training_seconds": time.perf_counter() - start,
        "history": history,
    }
    checkpoint_path.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
