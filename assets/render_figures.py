"""Render aggregate benchmark visuals from checked-in evidence; no rows or model fitting."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PAPER = "#f8fafc"
CARD = "#ffffff"
BORDER = "#d8e1ea"
INK = "#0f2537"
MUTED = "#486073"
TEAL = "#0f6a73"
TEAL_SOFT = "#e3f2f3"
AMBER = "#b85d19"
AMBER_SOFT = "#fdf0e4"
AMBER_INK = "#7c3a0a"
CRIMSON = "#a63a3a"
CRIMSON_SOFT = "#fbeaea"
CRIMSON_INK = "#7f2323"
STEEL = "#335c81"
SLATE_SOFT = "#e8eef5"
GRID = "#dde5ed"


def load_verified_benchmark(root: Path) -> tuple[dict, dict, int]:
    """Load and cross-check aggregate benchmark counts from checked-in JSON."""
    evidence = json.loads((root / "evidence/classifier-benchmark.json").read_text())
    result = evidence["evaluation"]
    counts = result["confusion_matrix"]
    scored = counts["tp"] + counts["tn"] + counts["fp"] + counts["fn"]
    assert scored == result["scorable_rows"]
    abstentions = result["input_rows"] - result["scorable_rows"]
    assert abstentions == evidence["unscored_rows_derived"]
    assert np.isclose(result["coverage"], result["scorable_rows"] / result["input_rows"])
    assert np.isclose(result["recall"], counts["tp"] / (counts["tp"] + counts["fn"]))
    assert np.isclose(result["precision"], counts["tp"] / (counts["tp"] + counts["fp"]))
    return evidence, result, abstentions


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11.5,
            "text.color": INK,
            "axes.edgecolor": BORDER,
            "axes.labelcolor": INK,
        }
    )


def render_benchmark(root: Path, result: dict, abstentions: int) -> Path:
    """Render scored-record confusion counts, metrics with 95% Wilson CIs, and coverage."""
    counts = result["confusion_matrix"]
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    scored = result["scorable_rows"]
    total = result["input_rows"]
    aps_total = tp + fn
    other_total = tn + fp
    flagged_total = tp + fp
    other_route_total = tn + fn

    figure = plt.figure(figsize=(13.2, 7.3), dpi=160, facecolor=PAPER)
    figure.patches.append(
        patches.Rectangle((0, 0.988), 1, 0.012, transform=figure.transFigure, color=TEAL, zorder=10)
    )

    figure.text(
        0.045,
        0.94,
        "ONE-USE EXTERNAL BENCHMARK  ·  SCANIA AIR PRESSURE SYSTEM (APS)",
        color=TEAL,
        fontsize=10,
        weight="bold",
    )
    figure.text(
        0.045,
        0.885,
        f"Frozen classifier performance on {scored:,} scored records",
        color=INK,
        fontsize=22,
        weight="bold",
    )
    figure.text(
        0.045,
        0.842,
        f"{total:,} external test records  ·  threshold frozen before single evaluation  ·  "
        f"{abstentions} input-quality abstentions excluded from scored metrics",
        color=MUTED,
        fontsize=11.2,
    )

    left_card = patches.FancyBboxPatch(
        (0.045, 0.195),
        0.445,
        0.605,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        transform=figure.transFigure,
        facecolor=CARD,
        edgecolor=BORDER,
        linewidth=1.1,
        zorder=1,
    )
    right_card = patches.FancyBboxPatch(
        (0.515, 0.195),
        0.44,
        0.605,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        transform=figure.transFigure,
        facecolor=CARD,
        edgecolor=BORDER,
        linewidth=1.1,
        zorder=1,
    )
    footer_card = patches.FancyBboxPatch(
        (0.045, 0.045),
        0.91,
        0.115,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        transform=figure.transFigure,
        facecolor="#eef3f8",
        edgecolor=BORDER,
        linewidth=1.0,
        zorder=1,
    )
    figure.patches.extend([left_card, right_card, footer_card])

    # Left card: discrete semantic confusion matrix inside card bounds
    ax_cm = figure.add_axes([0.185, 0.305, 0.285, 0.39], zorder=2)
    ax_cm.set_facecolor(CARD)
    ax_cm.set_xlim(0, 2)
    ax_cm.set_ylim(0, 2)

    figure.text(
        0.068,
        0.758,
        f"Scored-record confusion matrix (n = {scored:,})",
        color=INK,
        fontsize=13,
        weight="bold",
        zorder=3,
    )
    figure.text(
        0.068,
        0.728,
        "Recorded fault class versus frozen-threshold review route",
        color=MUTED,
        fontsize=10,
        zorder=3,
    )

    cells = [
        (0, 1, tn, "True negative", SLATE_SOFT, "#bccad8", INK, MUTED),
        (1, 1, fp, "False positive", AMBER_SOFT, AMBER, AMBER_INK, AMBER_INK),
        (0, 0, fn, "False negative", CRIMSON_SOFT, CRIMSON, CRIMSON_INK, CRIMSON_INK),
        (1, 0, tp, "True positive", TEAL, TEAL, "#ffffff", "#d8f1f3"),
    ]
    for col, row, count, role, fill, edge, text_color, sub_color in cells:
        tile = patches.FancyBboxPatch(
            (col + 0.055, row + 0.055),
            0.89,
            0.89,
            boxstyle="round,pad=0.01,rounding_size=0.06",
            facecolor=fill,
            edgecolor=edge,
            linewidth=1.6,
        )
        ax_cm.add_patch(tile)
        ax_cm.text(
            col + 0.5,
            row + 0.58,
            f"{count:,}",
            ha="center",
            va="center",
            fontsize=22,
            weight="bold",
            color=text_color,
        )
        ax_cm.text(
            col + 0.5,
            row + 0.26,
            role,
            ha="center",
            va="center",
            fontsize=9.8,
            weight="bold",
            color=sub_color,
        )

    ax_cm.set_xticks([0.5, 1.5])
    ax_cm.set_xticklabels(
        [
            f"Other-fault review\n(n = {other_route_total:,})",
            f"APS review\n(n = {flagged_total:,})",
        ],
        fontsize=9.6,
        weight="bold",
        color=INK,
    )
    ax_cm.set_yticks([0.5, 1.5])
    ax_cm.set_yticklabels(
        [
            f"APS-related fault\n(n = {aps_total:,})",
            f"Other recorded\nfault (n = {other_total:,})",
        ],
        fontsize=9.6,
        weight="bold",
        color=INK,
    )
    ax_cm.set_xlabel(
        "Predicted review route", fontsize=10.2, weight="bold", color=MUTED, labelpad=8
    )
    ax_cm.tick_params(length=0, pad=7)
    for spine in ax_cm.spines.values():
        spine.set_visible(False)

    # Right card: scored metrics + separated coverage bar
    ax_bar = figure.add_axes([0.65, 0.305, 0.28, 0.39], zorder=2)
    ax_bar.set_facecolor(CARD)

    figure.text(
        0.538,
        0.758,
        "Metrics with explicit denominators",
        color=INK,
        fontsize=13,
        weight="bold",
        zorder=3,
    )
    figure.text(
        0.538,
        0.728,
        "Whiskers mark 95% Wilson intervals on scored recall and precision",
        color=MUTED,
        fontsize=10,
        zorder=3,
    )

    y_positions = [3.2, 2.2, 1.2, 0.0]
    values = [
        result["recall"],
        result["precision"],
        result["average_precision"],
        result["coverage"],
    ]
    colors = [TEAL, AMBER, STEEL, "#5b748b"]
    labels = [
        f"Recall\n{tp} / {aps_total} scored APS",
        f"Precision\n{tp} / {flagged_total} APS flags",
        f"Average precision\n{scored:,} scored",
        f"Scoring coverage\n{scored:,} / {total:,} input",
    ]
    formatted = [
        f"{result['recall']:.2%}",
        f"{result['precision']:.2%}",
        f"{result['average_precision']:.4f}",
        f"{result['coverage']:.1%}",
    ]

    ax_bar.barh(y_positions, values, color=colors, height=0.52, zorder=3)

    for pos, key in [(3.2, "recall_interval_95"), (2.2, "precision_interval_95")]:
        low, high = result[key]
        ax_bar.plot([low, high], [pos, pos], color=INK, linewidth=2.2, zorder=4)
        ax_bar.plot([low, low], [pos - 0.11, pos + 0.11], color=INK, linewidth=2.0, zorder=4)
        ax_bar.plot([high, high], [pos - 0.11, pos + 0.11], color=INK, linewidth=2.0, zorder=4)

    ax_bar.axhline(0.62, color=BORDER, linestyle="--", linewidth=1.1, zorder=2)
    ax_bar.text(
        0.02,
        0.69,
        "Scored-record classification metrics (n = 15,872)",
        fontsize=8.4,
        color=MUTED,
        va="bottom",
    )
    ax_bar.text(
        0.02,
        0.54,
        "Input-population eligibility (n = 16,000)",
        fontsize=8.4,
        color=MUTED,
        va="top",
    )

    for pos, val, text in zip(y_positions, values, formatted, strict=True):
        offset = 0.045 if pos in (3.2, 2.2) else 0.03
        x_pos = max(val, result["recall_interval_95"][1] if pos == 3.2 else val)
        if pos == 2.2:
            x_pos = max(val, result["precision_interval_95"][1])
        ax_bar.text(
            x_pos + offset,
            pos,
            text,
            va="center",
            ha="left",
            fontsize=12,
            weight="bold",
            color=INK,
        )

    ax_bar.set_yticks(y_positions)
    ax_bar.set_yticklabels(labels, fontsize=9.6, weight="bold", color=INK)
    ax_bar.set_xlim(0, 1.25)
    ax_bar.set_ylim(-0.55, 3.75)
    ax_bar.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_bar.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9.5, color=MUTED)
    ax_bar.set_xlabel("Proportion / score", fontsize=10.2, weight="bold", color=MUTED, labelpad=8)
    ax_bar.grid(axis="x", color=GRID, linewidth=0.8, zorder=1)
    ax_bar.set_axisbelow(True)
    ax_bar.tick_params(length=0, pad=7)
    for spine in ax_bar.spines.values():
        spine.set_visible(False)

    # Footer strip kept comfortably inside card borders
    figure.text(
        0.064,
        0.118,
        f"Abstention boundary:  {abstentions} records ({abstentions / total:.1%} of {total:,}) "
        "exceeded the 47.06% missingness limit and routed to insufficient_evidence.",
        color=INK,
        fontsize=10.3,
        weight="bold",
        zorder=3,
    )
    figure.text(
        0.064,
        0.072,
        f"Review burden:  {fp} false positives and {fn} false negatives on scored records  ·  "
        "Negative labels are other component faults, not healthy trucks (no field or ROI claim).",
        color=MUTED,
        fontsize=9.6,
        zorder=3,
    )

    target = root / "assets/benchmark.png"
    figure.savefig(target, dpi=160, facecolor=PAPER)
    plt.close(figure)
    return target


def render_review_tradeoff(root: Path, result: dict, abstentions: int) -> Path:
    """Render APS flag composition (n=656) vs APS capture/misses (n=368) and full accounting."""
    counts = result["confusion_matrix"]
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    total = result["input_rows"]
    scored = result["scorable_rows"]
    flagged = tp + fp
    aps_scored = tp + fn
    other_routed = tn + fn

    tp_of_flags = tp / flagged
    fp_of_flags = fp / flagged
    tp_of_aps = tp / aps_scored
    fn_of_aps = fn / aps_scored
    flags_per_tp = flagged / tp

    figure = plt.figure(figsize=(13.2, 6.9), dpi=160, facecolor=PAPER)
    figure.patches.append(
        patches.Rectangle(
            (0, 0.988), 1, 0.012, transform=figure.transFigure, color=AMBER, zorder=10
        )
    )

    figure.text(
        0.045,
        0.938,
        "TRIAGE QUEUE TRADEOFF  ·  FROZEN OPERATING THRESHOLD",
        color=AMBER,
        fontsize=10,
        weight="bold",
    )
    figure.text(
        0.045,
        0.882,
        f"{tp_of_aps:.0%} of scored APS faults flagged, {fp_of_flags:.0%} of flags were other faults",
        color=INK,
        fontsize=20.5,
        weight="bold",
    )
    figure.text(
        0.045,
        0.838,
        f"Comparing queue composition ({flagged} APS flags) against APS capture ({aps_scored} scored APS records) "
        f"within the {scored:,} scored test cases",
        color=MUTED,
        fontsize=11.0,
    )

    left_card = patches.FancyBboxPatch(
        (0.045, 0.22),
        0.44,
        0.58,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        transform=figure.transFigure,
        facecolor=CARD,
        edgecolor=BORDER,
        linewidth=1.1,
        zorder=1,
    )
    right_card = patches.FancyBboxPatch(
        (0.515, 0.22),
        0.44,
        0.58,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        transform=figure.transFigure,
        facecolor=CARD,
        edgecolor=BORDER,
        linewidth=1.1,
        zorder=1,
    )
    bottom_card = patches.FancyBboxPatch(
        (0.045, 0.045),
        0.91,
        0.14,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        transform=figure.transFigure,
        facecolor="#eef3f8",
        edgecolor=BORDER,
        linewidth=1.0,
        zorder=1,
    )
    figure.patches.extend([left_card, right_card, bottom_card])

    # Left panel: Composition of 656 APS review flags
    figure.text(
        0.068,
        0.752,
        f"1. Inside the {flagged} APS review flags",
        color=INK,
        fontsize=14,
        weight="bold",
        zorder=3,
    )
    figure.text(
        0.068,
        0.718,
        f"Precision = {tp_of_flags:.2%}  ·  ~{flags_per_tp:.1f} flags reviewed per confirmed APS case",
        color=MUTED,
        fontsize=10.3,
        zorder=3,
    )

    ax_left = figure.add_axes([0.068, 0.465, 0.394, 0.21], zorder=2)
    ax_left.set_facecolor(CARD)
    ax_left.barh([0], [tp], color=TEAL, height=0.52, edgecolor=CARD, linewidth=2)
    ax_left.barh([0], [fp], left=[tp], color=AMBER, height=0.52, edgecolor=CARD, linewidth=2)
    ax_left.text(
        tp / 2,
        0,
        f"{tp}\n{tp_of_flags:.1%}",
        ha="center",
        va="center",
        color="#ffffff",
        fontsize=13.5,
        weight="bold",
    )
    ax_left.text(
        tp + fp / 2,
        0,
        f"{fp}\n{fp_of_flags:.1%}",
        ha="center",
        va="center",
        color="#ffffff",
        fontsize=13.5,
        weight="bold",
    )
    ax_left.text(
        0,
        0.37,
        "Confirmed APS (TP)",
        ha="left",
        va="bottom",
        color=TEAL,
        fontsize=10.2,
        weight="bold",
    )
    ax_left.text(
        flagged,
        0.37,
        "Other recorded fault (FP)",
        ha="right",
        va="bottom",
        color=AMBER,
        fontsize=10.2,
        weight="bold",
    )
    ax_left.text(0, -0.37, "0", ha="left", va="top", color=MUTED, fontsize=9.6)
    ax_left.text(
        flagged,
        -0.37,
        f"{flagged} APS review flags",
        ha="right",
        va="top",
        color=MUTED,
        fontsize=9.6,
    )
    ax_left.set_xlim(0, flagged)
    ax_left.set_ylim(-0.58, 0.62)
    ax_left.set_xticks([])
    ax_left.set_yticks([])
    for spine in ax_left.spines.values():
        spine.set_visible(False)

    left_box_tp = patches.FancyBboxPatch(
        (0.068, 0.335),
        0.394,
        0.088,
        boxstyle="round,pad=0.005,rounding_size=0.01",
        transform=figure.transFigure,
        facecolor=TEAL_SOFT,
        edgecolor="#b7dfe2",
        linewidth=0.9,
        zorder=2,
    )
    left_box_fp = patches.FancyBboxPatch(
        (0.068, 0.238),
        0.394,
        0.088,
        boxstyle="round,pad=0.005,rounding_size=0.01",
        transform=figure.transFigure,
        facecolor=AMBER_SOFT,
        edgecolor="#ebd0b8",
        linewidth=0.9,
        zorder=2,
    )
    figure.patches.extend([left_box_tp, left_box_fp])
    figure.text(
        0.082,
        0.383,
        f"{tp} confirmed APS faults",
        color=TEAL,
        fontsize=11.4,
        weight="bold",
        zorder=3,
    )
    figure.text(
        0.082,
        0.349,
        f"{tp_of_flags:.2%} of APS flags  ·  correctly routed to specialist review",
        color=INK,
        fontsize=9.5,
        zorder=3,
    )
    figure.text(
        0.082,
        0.286,
        f"{fp} other recorded component faults",
        color=AMBER_INK,
        fontsize=11.4,
        weight="bold",
        zorder=3,
    )
    figure.text(
        0.082,
        0.252,
        f"{fp_of_flags:.2%} of APS flags  ·  false-positive review burden",
        color=INK,
        fontsize=9.5,
        zorder=3,
    )

    # Right panel: Capture vs misses among 368 scored APS records
    figure.text(
        0.538,
        0.752,
        f"2. Fate of the {aps_scored} scored APS records",
        color=INK,
        fontsize=14,
        weight="bold",
        zorder=3,
    )
    figure.text(
        0.538,
        0.718,
        f"Recall = {tp_of_aps:.2%}  ·  threshold frozen on development data (target ≥ 85%)",
        color=MUTED,
        fontsize=10.3,
        zorder=3,
    )

    ax_right = figure.add_axes([0.538, 0.465, 0.394, 0.21], zorder=2)
    ax_right.set_facecolor(CARD)
    ax_right.barh([0], [tp], color=TEAL, height=0.52, edgecolor=CARD, linewidth=2)
    ax_right.barh([0], [fn], left=[tp], color=CRIMSON, height=0.52, edgecolor=CARD, linewidth=2)
    ax_right.text(
        tp / 2,
        0,
        f"{tp} routed\n{tp_of_aps:.2%}",
        ha="center",
        va="center",
        color="#ffffff",
        fontsize=13.5,
        weight="bold",
    )
    ax_right.text(
        0,
        0.37,
        "Captured for APS review (TP)",
        ha="left",
        va="bottom",
        color=TEAL,
        fontsize=10.2,
        weight="bold",
    )
    ax_right.annotate(
        f"{fn} missed ({fn_of_aps:.2%})",
        xy=(tp + fn / 2, 0.26),
        xytext=(aps_scored, 0.39),
        ha="right",
        va="bottom",
        fontsize=10.2,
        weight="bold",
        color=CRIMSON_INK,
        arrowprops={"arrowstyle": "->", "color": CRIMSON, "lw": 1.4, "shrinkA": 2, "shrinkB": 3},
    )
    ax_right.text(0, -0.37, "0", ha="left", va="top", color=MUTED, fontsize=9.6)
    ax_right.text(
        aps_scored,
        -0.37,
        f"{aps_scored} scored APS records",
        ha="right",
        va="top",
        color=MUTED,
        fontsize=9.6,
    )
    ax_right.set_xlim(0, aps_scored)
    ax_right.set_ylim(-0.58, 0.62)
    ax_right.set_xticks([])
    ax_right.set_yticks([])
    for spine in ax_right.spines.values():
        spine.set_visible(False)

    right_box_tp = patches.FancyBboxPatch(
        (0.538, 0.335),
        0.394,
        0.088,
        boxstyle="round,pad=0.005,rounding_size=0.01",
        transform=figure.transFigure,
        facecolor=TEAL_SOFT,
        edgecolor="#b7dfe2",
        linewidth=0.9,
        zorder=2,
    )
    right_box_fn = patches.FancyBboxPatch(
        (0.538, 0.238),
        0.394,
        0.088,
        boxstyle="round,pad=0.005,rounding_size=0.01",
        transform=figure.transFigure,
        facecolor=CRIMSON_SOFT,
        edgecolor="#e8c1c1",
        linewidth=0.9,
        zorder=2,
    )
    figure.patches.extend([right_box_tp, right_box_fn])
    figure.text(
        0.552,
        0.383,
        f"{tp} APS faults routed to APS review",
        color=TEAL,
        fontsize=11.4,
        weight="bold",
        zorder=3,
    )
    figure.text(
        0.552,
        0.349,
        f"{tp_of_aps:.2%} of scored APS records  ·  95% Wilson CI 91.1%–96.0%",
        color=INK,
        fontsize=9.5,
        zorder=3,
    )
    figure.text(
        0.552,
        0.286,
        f"{fn} APS faults missed to other-fault route",
        color=CRIMSON_INK,
        fontsize=11.4,
        weight="bold",
        zorder=3,
    )
    figure.text(
        0.552,
        0.252,
        f"{fn_of_aps:.2%} of scored APS records  ·  false-negative leakage",
        color=INK,
        fontsize=9.5,
        zorder=3,
    )

    # Bottom accounting band across all 16,000 external test inputs
    figure.text(
        0.064,
        0.145,
        f"Full {total:,}-record external-test routing accounting:",
        color=INK,
        fontsize=10.8,
        weight="bold",
        zorder=3,
    )
    figure.text(
        0.064,
        0.104,
        f"APS review: {flagged:,} ({flagged / total:.2%})    ·    "
        f"Other-fault review: {other_routed:,} ({other_routed / total:.2%})    ·    "
        f"Input-quality abstentions: {abstentions} ({abstentions / total:.2%})",
        color=INK,
        fontsize=10.4,
        zorder=3,
    )
    figure.text(
        0.064,
        0.064,
        f"{abstentions} abstentions sit outside both scored panels (missingness > 47.06%).  "
        "Benchmark error weights are comparison units, not financial savings.",
        color=MUTED,
        fontsize=9.5,
        zorder=3,
    )

    target = root / "assets/review-tradeoff.png"
    figure.savefig(target, dpi=160, facecolor=PAPER)
    plt.close(figure)
    return target


def render_social_preview(root: Path, result: dict, abstentions: int) -> Path:
    """Render 1280x640 repository preview matching the light editorial visual language."""
    counts = result["confusion_matrix"]
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    aps_total = tp + fn
    flagged_total = tp + fp
    scored = result["scorable_rows"]

    figure = plt.figure(figsize=(8, 4), dpi=160, facecolor=PAPER)
    figure.patches.append(
        patches.Rectangle((0, 0.975), 1, 0.025, transform=figure.transFigure, color=TEAL, zorder=10)
    )
    figure.patches.append(
        patches.Rectangle((0, 0), 0.014, 1, transform=figure.transFigure, color=INK, zorder=9)
    )

    figure.text(
        0.06,
        0.875,
        "ML ENGINEERING CASE STUDY  ·  AZURE DATABRICKS",
        color=TEAL,
        fontsize=9.2,
        weight="bold",
    )
    figure.text(
        0.06,
        0.725,
        "Fleet Fault Triage",
        color=INK,
        fontsize=28,
        weight="bold",
    )
    figure.text(
        0.06,
        0.625,
        "Air Pressure System fault routing  ·  MLflow  ·  Unity Catalog  ·  Delta persistence",
        color=MUTED,
        fontsize=11.2,
    )

    cards = [
        (f"{result['recall']:.2%}", "RECALL", f"{tp} / {aps_total} scored APS", TEAL),
        (f"{result['precision']:.2%}", "PRECISION", f"{tp} / {flagged_total} APS flags", AMBER),
        (f"{result['average_precision']:.4f}", "AVG PRECISION", f"{scored:,} scored rows", STEEL),
        (f"{result['coverage']:.1%}", "COVERAGE", f"{abstentions} abstentions", "#5b748b"),
    ]
    card_w = 0.205
    gap = 0.02
    start_x = 0.06
    for index, (value, label, detail, accent) in enumerate(cards):
        x = start_x + index * (card_w + gap)
        box = patches.FancyBboxPatch(
            (x, 0.255),
            card_w,
            0.295,
            boxstyle="round,pad=0.006,rounding_size=0.015",
            transform=figure.transFigure,
            facecolor=CARD,
            edgecolor=BORDER,
            linewidth=1.0,
            zorder=2,
        )
        bar = patches.Rectangle(
            (x + 0.008, 0.542),
            card_w - 0.016,
            0.012,
            transform=figure.transFigure,
            color=accent,
            zorder=3,
        )
        figure.patches.extend([box, bar])
        figure.text(x + 0.016, 0.425, value, color=INK, fontsize=18.5, weight="bold", zorder=4)
        figure.text(x + 0.016, 0.352, label, color=accent, fontsize=7.6, weight="bold", zorder=4)
        figure.text(x + 0.016, 0.285, detail, color=MUTED, fontsize=8.0, zorder=4)

    footer = patches.FancyBboxPatch(
        (0.06, 0.065),
        0.88,
        0.13,
        boxstyle="round,pad=0.005,rounding_size=0.012",
        transform=figure.transFigure,
        facecolor="#eef3f8",
        edgecolor=BORDER,
        linewidth=0.9,
        zorder=2,
    )
    figure.patches.append(footer)
    figure.text(
        0.08,
        0.138,
        "Frozen one-use Scania APS external benchmark  ·  metrics on 15,872 scored records",
        color=INK,
        fontsize=9.2,
        weight="bold",
        zorder=3,
    )
    figure.text(
        0.08,
        0.088,
        "11,910-row local/cloud parity (2.22e-16 max diff)  ·  inspectable source  ·  local synthetic demo",
        color=MUTED,
        fontsize=8.6,
        zorder=3,
    )

    target = root / "assets/social-preview.png"
    figure.savefig(target, dpi=160, facecolor=PAPER)
    plt.close(figure)
    return target


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    _, result, abstentions = load_verified_benchmark(root)
    _configure_style()
    outputs = [
        render_benchmark(root, result, abstentions),
        render_review_tradeoff(root, result, abstentions),
        render_social_preview(root, result, abstentions),
    ]
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
