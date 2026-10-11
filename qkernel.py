"""
qkernel.py - Quantum Kernel Classifier (Qiskit)

Pipeline:
    x  --feature map U(x)-->  |phi(x)>  ;   k(x, y) = |<phi(x)|phi(y)>|^2
    Gram matrix K  -->  classical SVM (kernel="precomputed")

The fidelity kernel is computed exactly from statevectors (fast, deterministic),
with optional shot-noise emulation and a full Aer noisy-circuit path.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import pauli_feature_map, zz_feature_map
from qiskit.quantum_info import Statevector
from scipy.linalg import expm
from sklearn.datasets import load_breast_cancer, make_circles, make_moons
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC


# --------------------------------------------------------------------------- #
# Feature maps
# --------------------------------------------------------------------------- #
def build_feature_map(name: str, n_qubits: int, reps: int = 2, entanglement: str = "linear") -> QuantumCircuit:
    """Return a parameterised feature-map circuit.

    name:
        "zz"  - ZZ feature map (H + P(x_i) + entangling ZZ phases). Hard to simulate classically.
        "z"   - product-state map (no entanglement). Classically simulable -> useful ablation.
        "xyz" - Pauli map with X, Y, ZZ terms (richer, deeper).
    """
    if name == "zz":
        return zz_feature_map(n_qubits, reps=reps, entanglement=entanglement)
    if name == "z":
        return pauli_feature_map(n_qubits, reps=reps, paulis=["Z"], entanglement=entanglement)
    if name == "xyz":
        return pauli_feature_map(n_qubits, reps=reps, paulis=["X", "Y", "ZZ"], entanglement=entanglement)
    raise ValueError(f"unknown feature map: {name}")


# --------------------------------------------------------------------------- #
# Kernel computation
# --------------------------------------------------------------------------- #
def encode_states(fmap: QuantumCircuit, X: np.ndarray) -> np.ndarray:
    """Statevectors |phi(x)> for every row of X -> array (n_samples, 2**n_qubits)."""
    return np.array([Statevector(fmap.assign_parameters(x)).data for x in X])


def fidelity_kernel(fmap: QuantumCircuit, XA: np.ndarray, XB: np.ndarray | None = None) -> np.ndarray:
    """Exact kernel K[i, j] = |<phi(A_i)|phi(B_j)>|^2."""
    SA = encode_states(fmap, XA)
    SB = SA if XB is None else encode_states(fmap, XB)
    return np.abs(SA.conj() @ SB.T) ** 2


def shot_noise_kernel(K_exact: np.ndarray, shots: int, rng: np.random.Generator, symmetric: bool = False) -> np.ndarray:
    """Emulate finite-shot estimation: each entry ~ Binomial(shots, k)/shots."""
    K = rng.binomial(shots, np.clip(K_exact, 0, 1)) / shots
    if symmetric:
        K = np.triu(K) + np.triu(K, 1).T
        np.fill_diagonal(K, 1.0)
    return K


def noisy_aer_kernel(fmap: QuantumCircuit, XA: np.ndarray, XB: np.ndarray | None = None,
                     shots: int = 1024, p1: float = 1e-3, p2: float = 1e-2, seed: int = 0) -> np.ndarray:
    """Kernel from compute-uncompute circuits U(y)^dagger U(x)|0> on a noisy Aer simulator.

    K[i, j] = P(measure all-zeros). Depolarising noise: p1 on 1q gates, p2 on 2q gates.
    """
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["u1", "u2", "u3", "rz", "sx", "x", "h", "p"])
    noise.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    sim = AerSimulator(noise_model=noise, seed_simulator=seed)
    basis = ["rz", "sx", "x", "cx"]

    symmetric = XB is None
    XB_ = XA if symmetric else XB
    n_a, n_b = len(XA), len(XB_)
    K = np.zeros((n_a, n_b))
    circuits, index = [], []
    for i in range(n_a):
        for j in range(n_b):
            if symmetric and j < i:
                continue
            if symmetric and i == j:
                K[i, j] = 1.0
                continue
            qc = fmap.assign_parameters(XA[i]).compose(fmap.assign_parameters(XB_[j]).inverse())
            qc.measure_all()
            circuits.append(transpile(qc, basis_gates=basis, optimization_level=1))
            index.append((i, j))
    # chunk to keep memory sane
    for s in range(0, len(circuits), 500):
        res = sim.run(circuits[s:s + 500], shots=shots).result()
        for k, (i, j) in enumerate(index[s:s + 500]):
            counts = res.get_counts(k)
            zeros = "0" * fmap.num_qubits
            K[i, j] = counts.get(zeros, 0) / shots
            if symmetric:
                K[j, i] = K[i, j]
    return K


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
def ad_hoc_dataset(n_per_class: int = 60, gap: float = 0.3, seed: int = 0):
    """Havlicek et al. (Nature 2019)-style dataset: labels are generated by a
    ZZ-feature-map expectation value, so the problem is natively 'quantum'.

    2 features in [0, 2pi). label = sign(<phi(x)| V^dag (Z x Z) V |phi(x)>), points
    inside |value| < gap are discarded to enforce a margin.
    """
    rng = np.random.default_rng(seed)
    fmap = zz_feature_map(2, reps=2, entanglement="linear")
    H = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    H = (H + H.conj().T) / 2
    V = expm(1j * H)
    ZZ = np.diag([1, -1, -1, 1]).astype(complex)
    obs = V.conj().T @ ZZ @ V

    pos, neg = [], []
    while min(len(pos), len(neg)) < n_per_class:
        x = rng.uniform(0, 2 * np.pi, size=2)
        psi = Statevector(fmap.assign_parameters(x)).data
        v = np.real(psi.conj() @ obs @ psi)
        if v > gap and len(pos) < n_per_class:
            pos.append(x)
        elif v < -gap and len(neg) < n_per_class:
            neg.append(x)
    X = np.vstack([pos, neg])
    y = np.array([1] * len(pos) + [0] * len(neg))
    return X, y


def load_dataset(name: str, n_samples: int = 200, n_features: int = 4, seed: int = 0):
    """Returns (X, y) with X unscaled. Binary labels in {0, 1}."""
    if name == "moons":
        return make_moons(n_samples=n_samples, noise=0.15, random_state=seed)
    if name == "circles":
        return make_circles(n_samples=n_samples, noise=0.1, factor=0.4, random_state=seed)
    if name == "adhoc":
        return ad_hoc_dataset(n_per_class=n_samples // 2, seed=seed)
    if name == "breast_cancer":
        X, y = load_breast_cancer(return_X_y=True)
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
        X, y = X[idx], y[idx]
        X = StandardScaler().fit_transform(X)
        X = PCA(n_components=n_features, random_state=seed).fit_transform(X)
        return X, y
    raise ValueError(f"unknown dataset: {name}")


def base_range(dataset: str):
    """Natural feature range for each dataset. 'adhoc' labels are generated on [0, 2pi)."""
    return (0.0, 2 * np.pi) if dataset == "adhoc" else (0.0, np.pi)


def prepare_split(X, y, test_size=0.3, seed=0, scale_range=(0, np.pi), scale_mult=1.0):
    """Stratified split, then MinMax-scale to scale_range fitted on TRAIN only (no leakage)."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    sc = MinMaxScaler(feature_range=scale_range).fit(Xtr)
    return sc.transform(Xtr) * scale_mult, sc.transform(Xte) * scale_mult, ytr, yte


# --------------------------------------------------------------------------- #
# Classifier
# --------------------------------------------------------------------------- #
def kernel_target_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """Alignment between K and the ideal kernel y y^T (labels mapped to +-1). Higher = better."""
    t = 2 * y - 1
    Y = np.outer(t, t)
    return float(np.sum(K * Y) / (np.linalg.norm(K) * np.linalg.norm(Y)))


class PrecomputedKernelSVM:
    """SVM on a precomputed Gram matrix with C chosen by stratified CV on the train kernel."""

    def __init__(self, C_grid=(0.1, 0.5, 1, 5, 10, 50, 100), cv=5, seed=0):
        self.C_grid, self.cv, self.seed = C_grid, cv, seed

    def fit(self, K_train, y_train):
        skf = StratifiedKFold(self.cv, shuffle=True, random_state=self.seed)
        best, best_C = -1, self.C_grid[0]
        for C in self.C_grid:
            accs = []
            for tr, va in skf.split(K_train, y_train):
                m = SVC(kernel="precomputed", C=C).fit(K_train[np.ix_(tr, tr)], y_train[tr])
                accs.append(accuracy_score(y_train[va], m.predict(K_train[np.ix_(va, tr)])))
            if np.mean(accs) > best:
                best, best_C = np.mean(accs), C
        self.C_, self.cv_acc_ = best_C, best
        self.model_ = SVC(kernel="precomputed", C=best_C).fit(K_train, y_train)
        return self

    def predict(self, K_test_train):
        return self.model_.predict(K_test_train)


def evaluate_quantum(fmap, Xtr, Xte, ytr, yte, seed=0, K_tr=None, K_te=None):
    K_tr = fidelity_kernel(fmap, Xtr) if K_tr is None else K_tr
    K_te = fidelity_kernel(fmap, Xte, Xtr) if K_te is None else K_te
    clf = PrecomputedKernelSVM(seed=seed).fit(K_tr, ytr)
    pred = clf.predict(K_te)
    return {
        "acc": accuracy_score(yte, pred),
        "f1": f1_score(yte, pred),
        "cv_acc": clf.cv_acc_,
        "C": clf.C_,
        "kta": kernel_target_alignment(K_tr, ytr),
    }


def evaluate_classical(kind, Xtr, Xte, ytr, yte, seed=0):
    """Tuned classical baselines for a fair comparison."""
    from sklearn.model_selection import GridSearchCV
    grids = {
        "rbf": (SVC(kernel="rbf"), {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.1, 1, 10]}),
        "linear": (SVC(kernel="linear"), {"C": [0.1, 1, 10, 100]}),
        "poly": (SVC(kernel="poly"), {"C": [0.1, 1, 10], "degree": [2, 3], "gamma": ["scale", 1]}),
    }
    est, grid = grids[kind]
    gs = GridSearchCV(est, grid, cv=StratifiedKFold(5, shuffle=True, random_state=seed)).fit(Xtr, ytr)
    pred = gs.predict(Xte)
    return {"acc": accuracy_score(yte, pred), "f1": f1_score(yte, pred), "cv_acc": gs.best_score_}


def evaluate_quantum_tuned(fmap_factory, Xtr, Xte, ytr, yte, mults=(0.1, 0.25, 0.5, 1.0), seed=0):
    """Tune kernel bandwidth (feature scale) by CV on the TRAIN set only, then evaluate on test.

    fmap_factory: callable() -> feature-map circuit.  Xtr/Xte already scaled to the base range.
    """
    fm = fmap_factory()
    best = None
    for m in mults:
        K_tr = fidelity_kernel(fm, Xtr * m)
        clf = PrecomputedKernelSVM(seed=seed).fit(K_tr, ytr)
        if best is None or clf.cv_acc_ > best[0]:
            best = (clf.cv_acc_, m)
    m = best[1]
    out = evaluate_quantum(fm, Xtr * m, Xte * m, ytr, yte, seed=seed)
    out["bandwidth"] = m
    return out
