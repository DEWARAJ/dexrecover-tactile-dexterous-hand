from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dexterous_hand_learning.config import ExperimentConfig
from dexterous_hand_learning.dagger import train_dagger
from dexterous_hand_learning.environment import HandEnvironment
from dexterous_hand_learning.policy import NormalizedPolicy
from dexterous_hand_learning.robust_evaluation import evaluate_robust


def main() -> None:
    config = ExperimentConfig.load(ROOT / "configs" / "default.json")
    output = ROOT / "artifacts" / "dagger"
    output.mkdir(parents=True, exist_ok=True)
    env = HandEnvironment(
        ROOT / "assets" / "four_finger_hand.xml",
        config.control_dt,
        config.episode_steps,
        config.success_radius_m,
        config.max_action_age_s,
    )
    base_checkpoint = ROOT / "artifacts" / "reference" / "behavior_cloning.pt"
    baseline = evaluate_robust(env, NormalizedPolicy.load(base_checkpoint), config)
    training = train_dagger(
        env,
        ROOT / "artifacts" / "reference" / "demonstrations.npz",
        base_checkpoint,
        output / "aggregated_demonstrations.npz",
        output / "dagger_policy.pt",
        config,
    )
    dagger = evaluate_robust(env, NormalizedPolicy.load(output / "dagger_policy.pt"), config)
    report = {"baseline_behavior_cloning": baseline, "dagger": dagger, "training": training}
    (output / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    labels = ["BC", "DAgger"]
    rates = [baseline["success_rate"] * 100, dagger["success_rate"] * 100]
    errors = [baseline["mean_final_distance_m"] * 1000, dagger["mean_final_distance_m"] * 1000]
    axes[0].bar(labels, rates, color=["#5975a4", "#1f9d72"])
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Success rate (%)")
    axes[0].set_title("100 robust randomized trials")
    axes[1].bar(labels, errors, color=["#5975a4", "#1f9d72"])
    axes[1].set_ylabel("Mean final error (mm)")
    axes[1].set_title("Lower is better")
    for axis in axes:
        axis.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Behavior cloning vs. DAgger")
    fig.tight_layout()
    fig.savefig(output / "bc_vs_dagger.png", dpi=180)
    plt.close(fig)
    print(json.dumps({
        "baseline": {k: v for k, v in baseline.items() if k != "trials"},
        "dagger": {k: v for k, v in dagger.items() if k != "trials"},
        "training": training,
    }, indent=2))


if __name__ == "__main__":
    main()
