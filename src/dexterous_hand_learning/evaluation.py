from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .environment import HandEnvironment
from .policy import NormalizedPolicy


def evaluate_policy(
    env: HandEnvironment,
    policy: NormalizedPolicy,
    config: ExperimentConfig,
    output_path: str | Path,
) -> dict:
    trials = []
    latencies_ms: list[float] = []
    for episode in range(config.evaluation_episodes):
        observation = env.reset(seed=10000 + config.seed + episode, randomize=True)
        info: dict[str, float | int | str | bool] = {"success": False, "distance_m": float("inf")}
        for step in range(config.episode_steps):
            phase = step / max(1, config.episode_steps - 1)
            started = time.perf_counter_ns()
            command = policy(observation)
            latencies_ms.append((time.perf_counter_ns() - started) / 1e6)
            result = env.step(command, phase)
            observation = result.observation
            info = result.info
        trials.append(
            {
                "episode": episode,
                "success": bool(info["success"]),
                "final_distance_m": float(info["distance_m"]),
                "safety_events": int(info["safety_events"]),
            }
        )

    successes = sum(int(t["success"]) for t in trials)
    report = {
        "episodes": config.evaluation_episodes,
        "success_radius_m": config.success_radius_m,
        "successes": successes,
        "success_rate": successes / config.evaluation_episodes,
        "mean_final_distance_m": float(np.mean([t["final_distance_m"] for t in trials])),
        "p95_final_distance_m": float(np.percentile([t["final_distance_m"] for t in trials], 95)),
        "mean_policy_latency_ms": float(np.mean(latencies_ms)),
        "p95_policy_latency_ms": float(np.percentile(latencies_ms, 95)),
        "total_safety_events": sum(int(t["safety_events"]) for t in trials),
        "trials": trials,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
