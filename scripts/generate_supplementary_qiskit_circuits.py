from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit import Gate


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "SpringerNature" / "sn-article-template" / "figs"


def _save_qiskit_figure(qc: QuantumCircuit, path: Path, *, title: str | None = None, scale: float = 0.9) -> None:
    fig = qc.draw(output="mpl", fold=-1, idle_wires=False, scale=scale)
    if title:
        fig.suptitle(title, y=1.02, fontsize=14, fontweight="bold")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _bell_probe_preparation() -> QuantumCircuit:
    qc = QuantumCircuit(2, name="Bell probe prep")
    qc.h(0)
    qc.cx(0, 1)
    return qc


def _correlator_measurement_circuit(label: str) -> QuantumCircuit:
    qc = QuantumCircuit(2, 2, name=label)
    qc.h(0)
    qc.cx(0, 1)
    qc.append(Gate(name="U_env", num_qubits=2, params=[]), [0, 1])

    left, right = label
    if left == "X":
        qc.h(0)
    elif left == "Y":
        qc.sdg(0)
        qc.h(0)

    if right == "X":
        qc.h(1)
    elif right == "Y":
        qc.sdg(1)
        qc.h(1)

    qc.measure([0, 1], [0, 1])
    return qc


def _compose_correlator_panel(path: Path) -> None:
    labels = ["XX", "YY", "XY", "YX"]
    with TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        temp_paths: list[Path] = []
        for label in labels:
            temp_path = tmp_dir / f"{label}.png"
            fig = _correlator_measurement_circuit(label).draw(output="mpl", fold=-1, idle_wires=False, scale=0.78)
            fig.savefig(temp_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            temp_paths.append(temp_path)

        panel, axes = plt.subplots(2, 2, figsize=(12.8, 5.8))
        panel.suptitle("B. Correlator measurement flow", fontsize=15, fontweight="bold", y=0.98)
        subtitle = (
            "The Bell probe is rotated into the required X/Y basis before measurement. "
            "The same readout pattern yields the XX, YY, XY, and YX correlators used by the latent estimator."
        )
        panel.text(0.5, 0.93, subtitle, ha="center", va="center", fontsize=10)

        for ax, label, temp_path in zip(axes.flatten(), labels, temp_paths):
            ax.imshow(mpimg.imread(temp_path))
            ax.set_title(label, fontsize=12.5, fontweight="bold", pad=8)
            ax.axis("off")

        panel.tight_layout(rect=[0, 0, 1, 0.90])
        panel.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
        panel.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(panel)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _save_qiskit_figure(
        _bell_probe_preparation(),
        OUT_DIR / "supp_fig6a_bell_probe_preparation",
        title="A. Bell-probe preparation",
        scale=1.0,
    )
    _compose_correlator_panel(OUT_DIR / "supp_fig6b_correlator_measurement_flow")
    print(f"saved circuit figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
