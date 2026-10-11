# Quantum Kernel Classifier (Qiskit)

A quantum-kernel SVM built with Qiskit, benchmarked against tuned classical baselines, with feature-map ablations and noise-robustness studies. Everything is reproducible from one command.

**Idea.** Encode a data point `x` into a quantum state with a feature map `U(x)|0>`. The kernel is the state overlap
`k(x, y) = |<phi(x)|phi(y)>|^2`. The resulting Gram matrix is passed to a classical SVM (`kernel="precomputed"`).

**Team (Qiskit Fall Fest 2026, SRM University-AP):** Sai Charan (Group Leader), Tanveer Khan, Doly Helan, Shaik Asif

## Run it in VS Code

**One-time setup** (Python 3.10 to 3.12 recommended)

1. Open this folder in VS Code (**File → Open Folder…**). Install the recommended extensions when prompted (Python, Pylance, Jupyter).
2. Open the terminal (**Terminal → New Terminal**) and create a virtual environment:

   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .venv\Scripts\Activate.ps1

   # macOS / Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```
   If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` once, then activate again.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Press **Ctrl+Shift+P → Python: Select Interpreter** and pick the one inside `.venv`.

**Run (pick any)**

| What | How |
|---|---|
| 30-second demo | Open `demo.py` and press the **Run ▶** button, or `python demo.py` |
| Step-by-step notebook | Open `quantum_kernel_classifier.ipynb`, choose the `.venv` kernel, **Run All** |
| Quick smoke test (~30 s) | **Run and Debug (Ctrl+Shift+D)** → "Run experiments (quick)" → ▶, or `python run_experiments.py --quick` (writes to `results_quick/`) |
| Full study (~6 min) | **Run and Debug** → "Run experiments (full, ~6 min)", or `python run_experiments.py` (writes to `results/`) |

The scripts find their own folder, so they work no matter which directory the terminal is in. The folder name contains spaces, so quote the path if you `cd` into it from a terminal: `cd "Quantum Kernel Classification"`.

**Troubleshooting**
- `ModuleNotFoundError: qiskit`: the wrong interpreter is selected. Re-do step 4, and make sure the terminal shows `(.venv)`.
- Circuit figure missing: `pip install pylatexenc` (already in `requirements.txt`).
- The full run feels slow: the Aer noisy-circuit study is the slow part (it simulates thousands of circuits). Use `--quick` while developing.

## Repo layout

| File | Purpose |
|---|---|
| `qkernel.py` | Feature maps (ZZ, Z, Pauli XYZ), exact fidelity kernel, shot-noise emulation, Aer noisy-circuit kernel, datasets, SVM wrapper with CV-tuned `C`, bandwidth tuning, kernel-target alignment |
| `run_experiments.py` | Benchmark, showcase plots, ablations, noise studies, auto-generated `results/summary.md` |
| `demo.py` | Short end-to-end demo for a live walkthrough |
| `quantum_kernel_classifier.ipynb` | Notebook walkthrough of the same pipeline |
| `results/` | All CSVs and figures from the full study |
| `.vscode/` | Run configurations and recommended extensions |

## Method

- **Feature maps:** ZZ feature map (entangling), Z map (no entanglement, classically simulable ablation), Pauli X/Y/ZZ map.
- **Kernel:** exact statevector fidelity for speed and determinism; also finite-shot emulation and a full Aer simulation of compute-uncompute circuits with depolarising gate noise.
- **Classifier:** `sklearn.svm.SVC(kernel="precomputed")`, `C` chosen by 5-fold stratified CV on the train kernel only.
- **Bandwidth tuning:** feature scale (kernel bandwidth) is tuned by CV on the training set only.
- **Fair baselines:** RBF, linear and polynomial SVMs, each with a CV grid search, on standardised features.
- **No leakage:** scaling is fitted on the train split only; hyper-parameters never see test data.
- **Datasets:** `moons`, `circles`, `breast_cancer` (PCA to 4 features, 4 qubits), and `adhoc` (a Havlicek et al.-style dataset whose labels come from a ZZ-feature-map observable, so it is quantum-native by construction).
- **Statistics:** 5 seeds per dataset, mean ± std.

## Results (test accuracy, mean ± std over 5 seeds)

| model | adhoc | breast_cancer | circles | moons |
|:--|:--|:--|:--|:--|
| Classical LINEAR | 0.553 ± 0.083 | 0.957 ± 0.040 | 0.540 ± 0.053 | 0.883 ± 0.020 |
| Classical POLY | 0.557 ± 0.056 | 0.913 ± 0.058 | 1.000 ± 0.000 | 0.880 ± 0.022 |
| Classical RBF | 0.800 ± 0.057 | 0.947 ± 0.034 | 0.997 ± 0.007 | 0.987 ± 0.014 |
| QK Z (no entanglement) | 0.747 ± 0.066 | 0.860 ± 0.040 | 1.000 ± 0.000 | 0.893 ± 0.030 |
| QK ZZ (reps=2, fixed scale) | 0.990 ± 0.015 | 0.677 ± 0.048 | 0.907 ± 0.049 | 0.787 ± 0.056 |
| **QK ZZ (CV-tuned bandwidth)** | **0.990 ± 0.015** | 0.900 ± 0.074 | 0.997 ± 0.007 | 0.953 ± 0.014 |

Figures: `results/fig_benchmark.png`, `fig_showcase_*.png` (kernel matrix vs ideal kernel + decision boundary), `fig_ablation_adhoc.png`, `fig_noise_shots.png`, `fig_noise_gates.png`, `fig_feature_map_circuit.png`.

### What the results say (honest reading)

1. **On quantum-native data the quantum kernel clearly wins** (0.99 vs 0.80 for the best classical kernel). This is expected: the labels are generated by the same kind of feature map. It demonstrates that the kernel is working correctly. It is **not** evidence of real-world quantum advantage.
2. **On standard classical datasets, the quantum kernel is competitive only after bandwidth tuning**, and classical RBF stays best or tied (moons 0.987 vs 0.953; breast cancer 0.947 vs 0.900). Without tuning, the ZZ kernel is poor (breast cancer 0.677), so bandwidth is the key hyper-parameter.
3. **Ablation (adhoc):** accuracy is highest (0.99) only when the feature map and the bandwidth match the data. Mismatched maps or scales fall to roughly 0.5-0.7, and deeper circuits (reps=3) do worse (0.81 at the native scale). The performance is therefore a feature-map-design problem, not a "more qubits or depth is better" problem.
4. **Noise robustness:** with finite shots (32 to 8192 per entry) accuracy stays at about 0.98 on adhoc after PSD projection of the noisy kernel. Under full Aer depolarising gate noise (50 train / 40 test, 3 seeds), mean accuracy goes from 0.81 (noiseless) to 0.77 at 10% two-qubit error. The variation across seeds (std about 0.15) is as large as the effect, so treat this as "mildly degrades", not a precise curve.

## Limitations

- Kernels are simulated classically (2 to 4 qubits). Exact evaluation scales as O(n^2) kernel entries and exponentially in qubits.
- Gate-noise study uses a simple depolarising model and small subsets because it needs O(n^2) circuits.
- No run on real IBM hardware here. Extension: swap the Aer backend for `qiskit-ibm-runtime` (`SamplerV2`) in `noisy_aer_kernel`, and use error mitigation.

## Possible extensions

Trainable feature maps (kernel-target alignment optimisation), projected quantum kernels, more qubits on MPS simulators, hardware run via IBM Quantum.
