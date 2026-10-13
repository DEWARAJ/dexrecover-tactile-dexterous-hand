from __future__ import annotations

import time

import numpy as np

from .config import ExperimentConfig
from .environment import HandEnvironment
from .policy import NormalizedPolicy


def evaluate_robust(
    env: HandEnvironment,
    policy: NormalizedPolicy,
    config: ExperimentConfig,
    episodes: int = 100,
    scale: float = 2.5,
    noise_std: float = 0.0015,
    action_delay_steps: int = 2,
) -> dict:
    rng = np.random.default_rng(config.seed + 9000)
    trials = []
    latencies = []
    for episode in range(episodes):
        observation = env.reset(
            seed=config.seed + 20000 + episode,
            randomize=True,
            randomization_scale=scale,
            domain_randomize=True,
        )
        delay = [env.data.ctrl.copy() for _ in range(action_delay_steps + 1)]
        info = {"success": False, "distance_m": float("inf"), "safety_events": 0}
        for step in range(config.episode_steps):
            phase = step / max(1, config.episode_steps - 1)
            noisy = observation + rng.normal(0.0, noise_std, observation.shape).astype(np.float32)
            noisy[-1] = phase
            started = time.perf_counter_ns()
            delay.append(policy(noisy))
            latencies.append((time.perf_counter_ns() - started) / 1e6)
            command = delay.pop(0)
            result = env.step(command, phase)
            observation = result.observation
            info = result.info
        trials.append({
            "success": bool(info["success"]),
            "final_distance_m": float(info["distance_m"]),
            "safety_events": int(info["safety_events"]),
        })
    distances = np.asarray([trial["final_distance_m"] for trial in trials])
    return {
        "episodes": episodes,
        "successes": int(sum(trial["success"] for trial in trials)),
        "success_rate": float(np.mean([trial["success"] for trial in trials])),
        "mean_final_distance_m": float(distances.mean()),
        "p95_final_distance_m": float(np.percentile(distances, 95)),
        "mean_policy_latency_ms": float(np.mean(latencies)),
        "total_safety_events": int(sum(trial["safety_events"] for trial in trials)),
        "randomization_scale": scale,
        "observation_noise_std": noise_std,
        "action_delay_steps": action_delay_steps,
        "trials": trials,
    }
