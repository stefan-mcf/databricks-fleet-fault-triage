"""Render aggregate benchmark evidence; no rows, model loading or fitting."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def render_cover(root, result):
    figure = plt.figure(figsize=(8, 4), dpi=160, facecolor="#102b3b")
    figure.text(0.07, 0.87, "ML ENGINEERING CASE STUDY", color="#77c5c7", fontsize=10)
    figure.text(0.07, 0.7, "Fleet Fault Triage", color="white", fontsize=31, weight="bold")
    figure.text(
        0.07, 0.59, "Databricks · MLflow · Unity Catalog · Delta", color="#d6e2e9", fontsize=13
    )
    values = [
        (f"{result['recall']:.2%}", "RECALL"),
        (f"{result['precision']:.2%}", "PRECISION"),
        (f"{result['average_precision']:.4f}", "AVERAGE PRECISION"),
        (f"{result['coverage']:.1%}", "COVERAGE"),
    ]
    for index, (value, label) in enumerate(values):
        x = 0.07 + 0.235 * index
        figure.text(x, 0.37, value, color="white", fontsize=20, weight="bold")
        figure.text(x, 0.29, label, color="#77c5c7", fontsize=7)
    figure.text(
        0.07,
        0.16,
        "Frozen Scania APS benchmark · metrics on scored records",
        color="#d6e2e9",
        fontsize=9,
    )
    figure.text(
        0.07,
        0.09,
        "Completed cloud jobs · inspectable source · local synthetic demo",
        color="#a7bdca",
        fontsize=8,
    )
    figure.savefig(root / "assets/social-preview.png", dpi=160, facecolor=figure.get_facecolor())
    plt.close(figure)


def main():
    root = Path(__file__).resolve().parents[1]
    evidence = json.loads((root / "evidence/classifier-benchmark.json").read_text())
    result = evidence["evaluation"]
    counts = result["confusion_matrix"]
    matrix = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]])
    assert int(matrix.sum()) == result["scorable_rows"]
    abstentions = result["input_rows"] - result["scorable_rows"]
    assert abstentions == evidence["unscored_rows_derived"]
    assert result["coverage"] == result["scorable_rows"] / result["input_rows"]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "text.color": "#172b3a"})
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(12.8, 6.7), gridspec_kw={"width_ratios": [1.05, 1]}
    )
    figure.patch.set_facecolor("#f6f8fa")
    figure.suptitle("Frozen external benchmark", x=0.07, ha="left", fontsize=23, weight="bold")
    figure.text(
        0.07,
        0.88,
        "Scania APS · recorded-fault classification · one-use 16,000-record test",
        fontsize=12,
    )
    left.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max())
    left.set_xticks([0, 1], ["Other-fault review", "APS review"])
    left.set_yticks([0, 1], ["Other recorded fault", "APS-related fault"])
    left.set_xlabel("Predicted route", labelpad=12)
    left.set_ylabel("Recorded class", labelpad=12)
    for row in range(2):
        for column in range(2):
            count = int(matrix[row, column])
            left.text(
                column,
                row,
                f"{count:,}",
                ha="center",
                va="center",
                fontsize=24,
                weight="bold",
                color="white" if count > matrix.max() / 2 else "#172b3a",
            )
    left.set_title("Confusion matrix · 15,872 scored records", fontsize=12, pad=16)
    left.tick_params(length=0, labelsize=10)
    for spine in left.spines.values():
        spine.set_visible(False)

    metrics = [result["recall"], result["precision"], result["average_precision"]]
    right.barh([2, 1, 0], metrics, color=["#176b78", "#b36a2f", "#3c668a"], height=0.5)
    right.set_yticks([2, 1, 0], ["Recall", "Precision", "Average precision"])
    right.set_xlim(0, 1.15)
    right.set_xticks([0, 0.5, 1.0])
    right.set_xlabel("Metric on scored records")
    right.set_facecolor("#f6f8fa")
    right.grid(axis="x", color="#dbe2e8", linewidth=0.7)
    right.set_axisbelow(True)
    right.set_title("High recall, material review burden", fontsize=12, pad=16)
    for position, value, label in zip(
        [2, 1, 0], metrics, [f"{metrics[0]:.2%}", f"{metrics[1]:.2%}", f"{metrics[2]:.4f}"]
    ):
        right.text(value + 0.025, position, label, va="center", weight="bold", fontsize=12)
    for spine in right.spines.values():
        spine.set_visible(False)
    right.tick_params(length=0, labelsize=10)
    figure.text(
        0.07,
        0.105,
        f"Coverage {result['coverage']:.1%} · {abstentions} input-quality abstentions excluded "
        "from scored-record metrics",
        fontsize=12,
        weight="bold",
    )
    figure.text(
        0.07,
        0.06,
        "310 false positives · 22 false negatives · no field accuracy or operational ROI claim",
        fontsize=11,
    )
    figure.subplots_adjust(left=0.16, right=0.94, top=0.78, bottom=0.24, wspace=0.8)
    target = root / "assets/benchmark.png"
    figure.savefig(target, dpi=160, facecolor=figure.get_facecolor())
    plt.close(figure)
    render_cover(root, result)
    print(target)


if __name__ == "__main__":
    main()
