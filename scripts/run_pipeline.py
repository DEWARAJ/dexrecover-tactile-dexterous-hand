from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dexterous_hand_learning.config import ExperimentConfig
from dexterous_hand_learning.dataset import collect_demonstrations
from dexterous_hand_learning.environment import HandEnvironment
from dexterous_hand_learning.evaluation import evaluate_policy
from dexterous_hand_learning.policy import NormalizedPolicy
from dexterous_hand_learning.reporting import create_plots
from dexterous_hand_learning.training import train_policy


def main() -> None:
    config = ExperimentConfig.load(ROOT / "configs" / "default.json")
    artifacts = ROOT / "artifacts" / "reference"
    dataset = artifacts / "demonstrations.npz"
    checkpoint = artifacts / "behavior_cloning.pt"
    evaluation = artifacts / "evaluation.json"

    env = HandEnvironment(
        ROOT / "assets" / "four_finger_hand.xml",
        config.control_dt,
        config.episode_steps,
        config.success_radius_m,
        config.max_action_age_s,
    )
    collection_report = collect_demonstrations(env, config, dataset)
    training_report = train_policy(dataset, checkpoint, config)
    policy = NormalizedPolicy.load(checkpoint)
    evaluation_report = evaluate_policy(env, policy, config, evaluation)
    plots = create_plots(checkpoint.with_suffix(".json"), evaluation, artifacts)

    summary = {
        "collection": collection_report,
        "training": {k: v for k, v in training_report.items() if k != "history"},
        "evaluation": {k: v for k, v in evaluation_report.items() if k != "trials"},
        "plots": [str(path.relative_to(ROOT)) for path in plots],
    }
    (artifacts / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

