from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

from src.utils.io import load_yaml


SERIES_COLORS = {
    "none": "#c84b55",
    "compensated": "#2f6690",
    "oracle": "#3f9b5f",
    "residual": "#d97b2d",
}

NOISE_COLORS = {
    "phi": "#5f6f7f",
    "gamma": "#7c6f9e",
    "amplitude": "#2a9d8f",
    "measurement": "#e76f51",
}

PANEL_BG = "#f6f8fb"
TEXT_DARK = "#17202a"
TEXT_MID = "#4c5b66"
GRID_FAINT = "#d6dde5"


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 10,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "savefig.facecolor": "white",
        }
    )


def _load_configs() -> dict[str, object]:
    config_paths = [
        ROOT / "configs" / "twobody" / "paper_figures_final.yaml",
        ROOT / "configs" / "twobody" / "paper_figures_final_3q.yaml",
        ROOT / "configs" / "twobody" / "paper_figures_final_4q.yaml",
    ]
    configs = [load_yaml(path) for path in config_paths]
    base = configs[0]
    n_values = [int(cfg["system"]["n_qubits"]) for cfg in configs]
    return {
        "n_values": n_values,
        "shots": int(base["experiment"]["shot_list"][0]),
        "train_seeds": len(base["experiment"]["train_seeds"]),
        "test_seeds": len(base["experiment"]["test_seeds"]),
        "probe_state_family": str(base["experiment"]["probe_state_family"]),
        "target_state_families": [str(x) for x in base["experiment"]["target_state_families"]],
        "feature_names": [str(x) for x in base["experiment"]["feature_names"]],
        "classifier": str(base["experiment"]["classifier"]),
        "system": dict(base["system"]),
        "probe_system": dict(base["probe_system"]),
        "decomposition": dict(base["decomposition"]),
        "transition": dict(base["transition"]),
    }


def _ensure_output_dir() -> Path:
    out_dir = ROOT / "DesignDocument" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=280, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _panel(ax, x: float, y: float, w: float, h: float, title: str, subtitle: str | None = None) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.0,
            edgecolor=GRID_FAINT,
            facecolor=PANEL_BG,
        )
    )
    ax.text(x + 0.02, y + h - 0.04, title, ha="left", va="top", fontsize=14, fontweight="bold", color=TEXT_DARK)
    if subtitle:
        ax.text(x + 0.02, y + h - 0.08, subtitle, ha="left", va="top", fontsize=10, color=TEXT_MID)


def _wire(ax, x0: float, x1: float, y: float, label: str) -> None:
    ax.plot([x0, x1], [y, y], color=TEXT_DARK, linewidth=1.8)
    ax.text(x0 - 0.02, y, label, ha="right", va="center", fontsize=10, color=TEXT_DARK)


def _gate(ax, x: float, y_center: float, w: float, h: float, text: str, face: str = "white", edge: str = TEXT_DARK) -> None:
    ax.add_patch(Rectangle((x, y_center - h / 2.0), w, h, linewidth=1.2, edgecolor=edge, facecolor=face))
    ax.text(x + w / 2.0, y_center, text, ha="center", va="center", fontsize=9.5, color=TEXT_DARK, fontweight="bold")


def _span_gate(
    ax,
    x: float,
    y_bottom: float,
    y_top: float,
    w: float,
    text: str,
    face: str = "white",
    edge: str = TEXT_DARK,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y_bottom),
            w,
            y_top - y_bottom,
            boxstyle="round,pad=0.008,rounding_size=0.015",
            linewidth=1.2,
            edgecolor=edge,
            facecolor=face,
        )
    )
    ax.text(x + w / 2.0, (y_bottom + y_top) / 2.0, text, ha="center", va="center", fontsize=10, color=TEXT_DARK)


def _cnot(ax, x: float, y_control: float, y_target: float) -> None:
    ax.add_patch(Circle((x, y_control), 0.008, color=TEXT_DARK))
    ax.plot([x, x], [y_target, y_control], color=TEXT_DARK, linewidth=1.3)
    ax.add_patch(Circle((x, y_target), 0.016, fill=False, edgecolor=TEXT_DARK, linewidth=1.2))
    ax.plot([x - 0.012, x + 0.012], [y_target, y_target], color=TEXT_DARK, linewidth=1.2)
    ax.plot([x, x], [y_target - 0.012, y_target + 0.012], color=TEXT_DARK, linewidth=1.2)


def _arrow(ax, start: tuple[float, float], end: tuple[float, float], color: str = TEXT_MID, lw: float = 1.6) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
        )
    )


def _label_chip(ax, x: float, y: float, text: str, color: str, w: float = 0.07) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            0.04,
            boxstyle="round,pad=0.006,rounding_size=0.012",
            linewidth=0,
            facecolor=color,
            alpha=0.96,
        )
    )
    ax.text(x + w / 2.0, y + 0.02, text, ha="center", va="center", fontsize=9, color="white", fontweight="bold")


def _text_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    lines: list[str],
    *,
    face: str = "white",
    edge: str = GRID_FAINT,
    title_color: str = TEXT_DARK,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=0.015",
            linewidth=1.0,
            edgecolor=edge,
            facecolor=face,
        )
    )
    ax.text(x + 0.018, y + h - 0.028, title, ha="left", va="top", fontsize=11.5, fontweight="bold", color=title_color)
    for idx, line in enumerate(lines):
        ax.text(x + 0.018, y + h - 0.066 - idx * 0.035, line, ha="left", va="top", fontsize=9.6, color=TEXT_MID)


def _floating_label(ax, x: float, y: float, text: str, *, fontsize: float = 9.8, weight: str = "bold") -> None:
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=TEXT_DARK,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 1.8},
    )


def _draw_quantum_circuit_figure(meta: dict[str, object], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14.5, 8.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.04, 0.965, "Latent-Neutrality Quantum Circuit Overview", ha="left", va="top", fontsize=18, fontweight="bold", color=TEXT_DARK)
    ax.text(
        0.04,
        0.93,
        "Separate system and Bell-probe branches share the same noise environment; the probe estimates the latent subspace used for compensation.",
        ha="left",
        va="top",
        fontsize=10.6,
        color=TEXT_MID,
    )

    _panel(ax, 0.035, 0.49, 0.93, 0.39, "A. System branch")
    _panel(ax, 0.035, 0.08, 0.93, 0.32, "B. Bell-probe branch")

    top_ys = [0.75, 0.68, 0.61, 0.54]
    for idx, y in enumerate(top_ys):
        label = f"q{idx}" if idx < 3 else "q(n-1)"
        _wire(ax, 0.08, 0.74, y, label)
    ax.text(0.082, 0.505, "n = 2, 3, 4 chain", ha="left", va="top", fontsize=9.2, color=TEXT_MID)

    _span_gate(ax, 0.12, 0.54, 0.80, 0.12, "", face="#ffffff")
    _gate(ax, 0.135, top_ys[0], 0.03, 0.045, "H")
    _cnot(ax, 0.175, top_ys[0], top_ys[1])
    _gate(ax, 0.195, top_ys[0], 0.03, 0.045, "S")
    _floating_label(ax, 0.18, 0.585, "State prep")

    evo = meta["system"]
    evo_text = f"U_XX+ZZ\nexact, t = {float(evo['evolution_time']):.1f}\nJx=1.0, Jz=0.5, hz=0.2"
    _span_gate(ax, 0.31, 0.54, 0.80, 0.12, "", face="#fdfdfd")
    _floating_label(ax, 0.37, 0.62, evo_text, fontsize=9.6)

    ax.add_patch(
        FancyBboxPatch(
            (0.46, 0.18),
            0.18,
            0.63,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.2,
            edgecolor="#d9e0e7",
            facecolor="#edf3f8",
            alpha=0.95,
        )
    )
    ax.text(0.55, 0.79, "Shared noise environment", ha="center", va="center", fontsize=12, fontweight="bold", color=TEXT_DARK)
    _label_chip(ax, 0.485, 0.735, "phi", NOISE_COLORS["phi"], w=0.055)
    _label_chip(ax, 0.547, 0.735, "gamma", NOISE_COLORS["gamma"], w=0.072)
    _label_chip(ax, 0.625, 0.735, "eta", NOISE_COLORS["amplitude"], w=0.055)
    _label_chip(ax, 0.518, 0.69, "p_meas", NOISE_COLORS["measurement"], w=0.09)

    _span_gate(ax, 0.68, 0.54, 0.80, 0.11, "", face="#ffffff")
    _floating_label(ax, 0.735, 0.67, "Observable set\nZi, ZiZj, XiXj, YiYj,\nXiYj / YiXj", fontsize=9.2)
    _arrow(ax, (0.24, 0.69), (0.31, 0.69))
    _arrow(ax, (0.43, 0.69), (0.46, 0.69))
    _arrow(ax, (0.64, 0.69), (0.68, 0.69))

    _text_box(
        ax,
        0.80,
        0.54,
        0.16,
        0.13,
        "Observable compensation",
        [
            "Undo phi frame rotation",
            "Divide by (1-gamma)^(n_perp / 2)",
        ],
        face="#eef5fb",
        edge=SERIES_COLORS["compensated"],
        title_color=SERIES_COLORS["compensated"],
    )
    _text_box(
        ax,
        0.80,
        0.72,
        0.16,
        0.11,
        "Feature + classifier",
        [
            "6-D feature vector",
            "Logistic boundary",
            "decision outputs",
        ],
        face="#ffffff",
    )
    _arrow(ax, (0.79, 0.69), (0.80, 0.64))
    _arrow(ax, (0.88, 0.67), (0.88, 0.72))

    probe_ys = [0.28, 0.20]
    for idx, y in enumerate(probe_ys):
        _wire(ax, 0.08, 0.74, y, f"p{idx}")
    ax.text(0.08, 0.295, "Latent estimation from XX / YY / XY / YX correlators", ha="left", va="bottom", fontsize=9.6, color=TEXT_MID)

    _span_gate(ax, 0.12, 0.17, 0.31, 0.12, "", face="#ffffff")
    _gate(ax, 0.14, probe_ys[0], 0.03, 0.045, "H")
    _cnot(ax, 0.19, probe_ys[0], probe_ys[1])
    _floating_label(ax, 0.18, 0.245, "Bell probe prep", fontsize=9.6)
    probe_cfg = meta["probe_system"]
    _span_gate(ax, 0.31, 0.17, 0.31, 0.10, "", face="#fdfdfd")
    _floating_label(ax, 0.37, 0.245, f"Probe evolution\nexact, t = {float(probe_cfg['evolution_time']):.1f}", fontsize=9.6)
    _span_gate(ax, 0.68, 0.17, 0.31, 0.11, "", face="#ffffff")
    _floating_label(ax, 0.735, 0.245, "Probe correlators\nXX, YY, XY, YX", fontsize=9.4)
    _arrow(ax, (0.24, 0.24), (0.31, 0.24))
    _arrow(ax, (0.41, 0.24), (0.46, 0.24))
    _arrow(ax, (0.64, 0.24), (0.68, 0.24))

    _text_box(
        ax,
        0.80,
        0.165,
        0.16,
        0.185,
        "Latent estimator",
        [
            "c_phi = (XX - YY) / 2",
            "s_phi = (XY + YX) / 2",
            "phi_hat = atan2(s_phi, c_phi)",
            "gamma_hat = 1 - sqrt(c_phi^2 + s_phi^2)",
        ],
        face="#f8f3fc",
        edge=NOISE_COLORS["gamma"],
        title_color=NOISE_COLORS["gamma"],
    )
    _arrow(ax, (0.79, 0.24), (0.80, 0.26))
    _arrow(ax, (0.88, 0.35), (0.88, 0.54), color=NOISE_COLORS["gamma"], lw=1.8)

    ax.text(
        0.04,
        0.03,
        "Current paper runs use the same noise configuration for probe and target circuits, 8192 shots, and the final compensated policy operates only in the phi / gamma latent subspace.",
        ha="left",
        va="bottom",
        fontsize=9.6,
        color=TEXT_MID,
    )

    _save(fig, out_dir / "latent_neutrality_quantum_circuit")


def _draw_protocol_figure(meta: dict[str, object], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14.5, 8.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.04, 0.965, "Experimental Protocol and Environment Schematic", ha="left", va="top", fontsize=18, fontweight="bold", color=TEXT_DARK)
    ax.text(
        0.04,
        0.93,
        "For this project, the environment definition is part of the claim: backend, shot count, seed averaging, and the shared probe/system noise model determine what the compensation policy can and cannot explain.",
        ha="left",
        va="top",
        fontsize=10.6,
        color=TEXT_MID,
    )

    _panel(ax, 0.035, 0.54, 0.27, 0.33, "1. Fixed experiment settings")
    _panel(ax, 0.34, 0.54, 0.30, 0.33, "2. Evaluation protocol")
    _panel(ax, 0.67, 0.54, 0.295, 0.33, "3. Outputs and interpretation")
    _panel(ax, 0.035, 0.08, 0.93, 0.38, "4. Why the environment matters")

    n_values = meta["n_values"]
    target_state_families = meta["target_state_families"]
    feature_names = meta["feature_names"]
    decomposition = meta["decomposition"]
    transition = meta["transition"]

    _text_box(
        ax,
        0.055,
        0.57,
        0.23,
        0.24,
        "Common runtime configuration",
        [
            f"Qubit counts: {', '.join(str(v) for v in n_values)}",
            f"Backend: shot, shots = {meta['shots']}",
            f"Train seeds = {meta['train_seeds']}, test seeds = {meta['test_seeds']}",
            f"Probe state = {meta['probe_state_family']}",
            f"Targets = {target_state_families[0]}, {target_state_families[1]}",
            "6-D decision vector",
            f"Classifier = {meta['classifier']}",
        ],
    )

    step_boxes = [
        (0.365, 0.70, "Train clean/full-oracle classifier", ["Learn the decision boundary", "before noisy evaluation"]),
        (0.365, 0.57, "Run probe and target under shared noise", ["Same phi / gamma / eta / p_meas", "applied to both branches"]),
        (0.365, 0.44, "Estimate latent and compare policies", ["none / compensated", "structured oracle / full oracle"]),
    ]
    for x, y, title, lines in step_boxes:
        _text_box(ax, x, y, 0.25, 0.10, title, lines, face="white")
    _arrow(ax, (0.49, 0.70), (0.49, 0.67))
    _arrow(ax, (0.49, 0.57), (0.49, 0.54))

    _text_box(
        ax,
        0.69,
        0.67,
        0.255,
        0.14,
        "Primary metrics",
        [
            "latent MAE",
            "observable MAE",
            "balanced accuracy / control gain",
            "oracle gap and fingerprint distance",
        ],
        face="#ffffff",
    )
    _text_box(
        ax,
        0.69,
        0.47,
        0.255,
        0.14,
        "Current narrative frame",
        [
            "2Q: strong MAE gain",
            "3Q/4Q: stable compensated recoverability",
            "4Q: selective composite fragility",
        ],
        face="#f6fbf7",
        edge=SERIES_COLORS["oracle"],
        title_color=SERIES_COLORS["oracle"],
    )

    _text_box(
        ax,
        0.06,
        0.12,
        0.27,
        0.22,
        "Noise axes actually swept",
        [
            f"phi up to {max(float(x) for x in decomposition['component_axes']['phi']):.2f}",
            f"gamma up to {max(float(x) for x in decomposition['component_axes']['gamma_dephasing']):.2f}",
            f"eta up to {max(float(x) for x in decomposition['component_axes']['eta_amplitude']):.2f}",
            f"p_meas up to {max(float(x) for x in decomposition['component_axes']['p_measurement']):.2f}",
            f"Transition axis = {transition['noise_axis']}",
            "Composite sweep mixes all four components",
        ],
        face="white",
    )

    _text_box(
        ax,
        0.365,
        0.12,
        0.27,
        0.22,
        "Interpretation constraint",
        [
            "Compensation only models phi and gamma.",
            "Amplitude and measurement alter observed",
            "populations and readout geometry outside",
            "the latent control family.",
            "This is why full oracle can stay high while",
            "structured compensation saturates.",
        ],
        face="#fff8f2",
        edge=NOISE_COLORS["measurement"],
        title_color=NOISE_COLORS["measurement"],
    )

    _text_box(
        ax,
        0.67,
        0.12,
        0.275,
        0.22,
        "Reviewer-facing value of this schematic",
        [
            "It fixes the experimental contract in one place:",
            "same backend, same shot budget, same seed averaging,",
            "same shared-noise probe/system pair, and explicit",
            "separation between structured oracle and full oracle.",
            "That makes the collapse claims auditable.",
        ],
        face="white",
    )


    _save(fig, out_dir / "latent_neutrality_protocol_schematic")


def main() -> None:
    _configure_matplotlib()
    meta = _load_configs()
    out_dir = _ensure_output_dir()
    _draw_quantum_circuit_figure(meta, out_dir)
    _draw_protocol_figure(meta, out_dir)
    print(f"saved diagrams to {out_dir}")


if __name__ == "__main__":
    main()
