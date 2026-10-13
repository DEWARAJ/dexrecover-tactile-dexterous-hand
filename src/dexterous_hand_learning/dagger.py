from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .environment import HandEnvironment
from .expert import scripted_expert
from .policy import NormalizedPolicy
from .training import train_policy


def rollout_dataset(
    env: HandEnvironment,
    policy: NormalizedPolicy,
    config: ExperimentConfig,
    episodes: int,
    episode_offset: int,
    scale: float,
    noise_std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observations: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    episode_ids: list[int] = []
    rng = np.random.default_rng(config.seed + 5000 + episode_offset)
    for local_episode in range(episodes):
        episode_id = episode_offset + local_episode
        observation = env.reset(
            seed=config.seed + 1000 + episode_id,
            randomize=True,
            randomization_scale=scale,
            domain_randomize=True,
        )
        start_xy = env.cube_position()[:2].copy()
        target_xy = env.target_xy.copy()
        for step in range(config.episode_steps):
            phase = step / max(1, config.episode_steps - 1)
            noisy_observation = observation + rng.normal(0.0, noise_std, observation.shape).astype(np.float32)
            noisy_observation[-1] = phase
            command = policy(noisy_observation)
            label = scripted_expert(step, config.episode_steps, start_xy, target_xy)
            observations.append(noisy_observation)
            labels.append(label.astype(np.float32))
            episode_ids.append(episode_id)
            observation = env.step(command, phase).observation
    return np.stack(observations), np.stack(labels), np.asarray(episode_ids, dtype=np.int32)


def train_dagger(
    env: HandEnvironment,
    base_dataset_path: str | Path,
    base_checkpoint_path: str | Path,
    output_dataset_path: str | Path,
    output_checkpoint_path: str | Path,
    config: ExperimentConfig,
    iterations: int = 2,
    episodes_per_iteration: int = 20,
    scale: float = 2.5,
    noise_std: float = 0.0015,
) -> dict:
    base = np.load(base_dataset_path)
    observations = [base["observations"].astype(np.float32)]
    actions = [base["actions"].astype(np.float32)]
    episode_ids = [base["episode_ids"].astype(np.int32)]
    policy = NormalizedPolicy.load(base_checkpoint_path)
    reports = []
    total_episodes = config.demonstrations
    for iteration in range(iterations):
        rollout_obs, rollout_labels, rollout_ids = rollout_dataset(
            env,
            policy,
            config,
            episodes_per_iteration,
            total_episodes,
            scale,
            noise_std,
        )
        observations.append(rollout_obs)
        actions.append(rollout_labels)
        episode_ids.append(rollout_ids)
        total_episodes += episodes_per_iteration
        output_dataset_path = Path(output_dataset_path)
        output_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_dataset_path,
            observations=np.concatenate(observations),
            actions=np.concatenate(actions),
            episode_ids=np.concatenate(episode_ids),
        )
        iteration_config = replace(config, demonstrations=total_episodes, train_epochs=80)
        report = train_policy(output_dataset_path, output_checkpoint_path, iteration_config)
        reports.append({k: value for k, value in report.items() if k != "history"})
        policy = NormalizedPolicy.load(output_checkpoint_path)
    return {"iterations": iterations, "episodes_per_iteration": episodes_per_iteration, "reports": reports}
