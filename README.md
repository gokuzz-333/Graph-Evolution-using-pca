# Hybrid Classical and Quantum Kernel PCA for Temporal Email Networks

This repository contains the project source, tests, experiment outputs, figures, and the PCA results workbook. The quantum-kernel experiment uses a Qiskit Statevector simulation; it is not a quantum-hardware result.

## Run

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_project.py
python -m pytest
```

The pipeline downloads the SNAP Email-Eu-core-temporal dataset if `data/email-Eu-core-temporal.txt.gz` is not present. The raw dataset is excluded from this repository.

## Project contents

- `run_project.py` and `src/`: pipeline, temporal graph construction, features, custom PCA, graph-matrix PCA, simulated quantum-kernel PCA, and visualizations.
- `tests/`: unit and pipeline tests.
- `results/`: saved CSV, NPZ, and JSON experiment outputs.
- `figures/`: plots generated from the saved run.
- `docs/`: code guide, requirements coverage, and viva notes.
- `PCA_Analysis_Results.xlsx`: labeled PCA results workbook.
