from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .environment import HandEnvironment
from .expert import scripted_expert


def collect_demonstrations(
    env: HandEnvironment,
    config: ExperimentConfig,
    output_path: str | Path,
) -> dict[str, float | int]:
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    episode_ids: list[int] = []
    successes = 0
    final_distances: list[float] = []

    for episode in range(config.demonstrations):
        observation = env.reset(seed=config.seed + episode, randomize=True)
        cube_xy = env.cube_position()[:2].copy()
        target_xy = env.target_xy.copy()
        info: dict[str, float | int | str | bool] = {"success": False, "distance_m": float("inf")}
        for step in range(config.episode_steps):
            phase = step / max(1, config.episode_steps - 1)
            action = scripted_expert(step, config.episode_steps, cube_xy, target_xy)
            observations.append(observation)
            actions.append(action.astype(np.float32))
            episode_ids.append(episode)
            result = env.step(action, phase)
            observation = result.observation
            info = result.info
        successes += int(bool(info["success"]))
        final_distances.append(float(info["distance_m"]))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        observations=np.stack(observations),
        actions=np.stack(actions),
        episode_ids=np.asarray(episode_ids, dtype=np.int32),
    )
    report = {
        "episodes": config.demonstrations,
        "samples": len(observations),
        "expert_successes": successes,
        "expert_success_rate": successes / config.demonstrations,
        "mean_final_distance_m": float(np.mean(final_distances)),
        "max_final_distance_m": float(np.max(final_distances)),
    }
    output_path.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

