from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dexterous_hand_learning.articulated import AllegroHandEnvironment, OPEN_POSE
from dexterous_hand_learning.tactile import NOMINAL_GRASP_POSE, TactileRecoveryController, estimate_tactile_state


def run_trial(env: AllegroHandEnvironment, seed: int, recovery: bool) -> dict:
    rng = np.random.default_rng(seed)
    env.reset(seed=seed, domain_randomize=True)
    for step in range(120):
        phase = step / 119
        blend = phase * phase * (3.0 - 2.0 * phase)
        env.step((1.0 - blend) * OPEN_POSE + blend * NOMINAL_GRASP_POSE)
    start = env.data.xpos[env.object_body_id].copy()
    controller = TactileRecoveryController(env)
    direction = rng.normal(size=2)
    direction /= np.linalg.norm(direction)
    force_n = rng.uniform(0.20, 0.45)
    slip_events = 0
    peak_speed = 0.0
    peak_force = 0.0
    for step in range(75):
        if step < 25:
            env.data.xfrc_applied[env.object_body_id, :2] = force_n * direction
        else:
            env.data.xfrc_applied[env.object_body_id] = 0.0
        tactile = estimate_tactile_state(env)
        slip_events += int(tactile.slip_detected)
        peak_speed = max(peak_speed, tactile.object_speed_mps)
        peak_force = max(peak_force, tactile.normal_force_n)
        command = controller.update(tactile) if recovery else NOMINAL_GRASP_POSE
        env.step(command)
    env.data.xfrc_applied[env.object_body_id] = 0.0
    tactile = estimate_tactile_state(env)
    displacement = float(np.linalg.norm(env.data.xpos[env.object_body_id] - start))
    success = tactile.contact_count >= 2 and displacement < 0.010
    return {
        "success": success,
        "final_contacts": tactile.contact_count,
        "displacement_m": displacement,
        "slip_events": slip_events,
        "peak_object_speed_mps": peak_speed,
        "peak_normal_force_n": peak_force,
        "disturbance_force_n": force_n,
    }


def summarize(trials: list[dict]) -> dict:
    return {
        "episodes": len(trials),
        "successes": int(sum(t["success"] for t in trials)),
        "success_rate": float(np.mean([t["success"] for t in trials])),
        "mean_displacement_m": float(np.mean([t["displacement_m"] for t in trials])),
        "p95_displacement_m": float(np.percentile([t["displacement_m"] for t in trials], 95)),
        "mean_slip_events": float(np.mean([t["slip_events"] for t in trials])),
        "trials": trials,
    }


def main() -> None:
    scene = ROOT / "third_party" / "wonik_allegro" / "scene_right.xml"
    baseline_env = AllegroHandEnvironment(scene)
    recovery_env = AllegroHandEnvironment(scene)
    seeds = [40000 + index for index in range(100)]
    baseline = summarize([run_trial(baseline_env, seed, False) for seed in seeds])
    recovery = summarize([run_trial(recovery_env, seed, True) for seed in seeds])
    report = {
        "baseline_fixed_grasp": baseline,
        "tactile_recovery": recovery,
        "success_criterion": ">=2 final hand-object contacts and <10 mm displacement after a 0.20-0.45 N lateral disturbance",
    }
    output = ROOT / "artifacts" / "tactile_recovery"
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    labels = ["Fixed grasp", "Tactile recovery"]
    rates = [baseline["success_rate"] * 100, recovery["success_rate"] * 100]
    displacement = [baseline["mean_displacement_m"] * 1000, recovery["mean_displacement_m"] * 1000]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    axes[0].bar(labels, rates, color=["#d87757", "#1f9d72"])
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Recovery success (%)")
    axes[0].set_title("Randomized disturbance trials")
    axes[1].bar(labels, displacement, color=["#d87757", "#1f9d72"])
    axes[1].set_ylabel("Mean object displacement (mm)")
    axes[1].set_title("Lower is better")
    for axis in axes:
        axis.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Tactile slip detection and grasp recovery")
    fig.tight_layout()
    fig.savefig(output / "tactile_recovery.png", dpi=180)
    plt.close(fig)
    print(json.dumps({
        "baseline": {k: v for k, v in baseline.items() if k != "trials"},
        "recovery": {k: v for k, v in recovery.items() if k != "trials"},
    }, indent=2))


if __name__ == "__main__":
    main()
