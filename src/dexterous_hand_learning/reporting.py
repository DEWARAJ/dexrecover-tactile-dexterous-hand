from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def create_plots(training_json: str | Path, evaluation_json: str | Path, output_dir: str | Path) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training = json.loads(Path(training_json).read_text(encoding="utf-8"))
    evaluation = json.loads(Path(evaluation_json).read_text(encoding="utf-8"))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(training["history"]["train_loss"], label="Train MSE", linewidth=2)
    ax.plot(training["history"]["validation_loss"], label="Validation MSE", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean-squared error")
    ax.set_yscale("log")
    ax.set_title("Behavior-cloning learning curve")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    learning = output_dir / "learning_curve.png"
    fig.savefig(learning, dpi=170)
    plt.close(fig)

    distances = np.array([x["final_distance_m"] for x in evaluation["trials"]]) * 1000
    success = np.array([x["success"] for x in evaluation["trials"]])
    colors = np.where(success, "#1f9d72", "#d95f4f")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(np.arange(distances.size), distances, color=colors)
    threshold_mm = float(evaluation["success_radius_m"]) * 1000
    ax.axhline(threshold_mm, color="#172a3a", linestyle="--", label=f"{threshold_mm:g} mm success radius")
    ax.set_xlabel("Unseen randomized trial")
    ax.set_ylabel("Final object-target distance (mm)")
    ax.set_title("Closed-loop policy evaluation")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    trials = output_dir / "evaluation_trials.png"
    fig.savefig(trials, dpi=170)
    plt.close(fig)
    return [learning, trials]
