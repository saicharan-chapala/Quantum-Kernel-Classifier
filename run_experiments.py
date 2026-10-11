"""
run_experiments.py - reproduces every result/plot for the Quantum Kernel Classifier project.

    python run_experiments.py            # full run (a few minutes on a laptop)
    python run_experiments.py --quick    # smoke test (< 1 minute) -> results_quick/

Outputs -> <this folder>/results/  (CSV tables, PNG figures, summary.md)
"""
import argparse
import os
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import qkernel as qk

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")   # always next to this file, whatever the working directory
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.25})


def psd_project(K):
    """Clip negative eigenvalues (needed for shot-noise / hardware kernels)."""
    K = (K + K.T) / 2
    w, V = np.linalg.eigh(K)
    return (V * np.clip(w, 0, None)) @ V.T


# --------------------------------------------------------------------------- #
# 1. Main benchmark: quantum kernels vs tuned classical baselines
# --------------------------------------------------------------------------- #
def benchmark(datasets, seeds, n_samples):
    rows = []
    for ds in datasets:
        n_feat = 2 if ds in ("moons", "circles", "adhoc") else 4
        for seed in seeds:
            X, y = qk.load_dataset(ds, n_samples=n_samples, n_features=n_feat, seed=seed)
            Xtr, Xte, ytr, yte = qk.prepare_split(X, y, seed=seed, scale_range=qk.base_range(ds))
            nq = Xtr.shape[1]
            models = {
                "QK ZZ (reps=2)": qk.build_feature_map("zz", nq, reps=2),
                "QK Z (no entanglement)": qk.build_feature_map("z", nq, reps=2),
            }
            for name, fm in models.items():
                r = qk.evaluate_quantum(fm, Xtr, Xte, ytr, yte, seed=seed)
                rows.append({"dataset": ds, "model": name, "seed": seed, **r})
            r = qk.evaluate_quantum_tuned(lambda: qk.build_feature_map("zz", nq, reps=2), Xtr, Xte, ytr, yte, seed=seed)
            rows.append({"dataset": ds, "model": "QK ZZ (CV-tuned bandwidth)", "seed": seed, **r})
            # classical baselines use standardised features (standard practice)
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler().fit(Xtr)
            for kind in ("rbf", "linear", "poly"):
                r = qk.evaluate_classical(kind, sc.transform(Xtr), sc.transform(Xte), ytr, yte, seed=seed)
                rows.append({"dataset": ds, "model": f"Classical {kind.upper()}", "seed": seed, **r})
        print(f"  [benchmark] {ds} done")
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/benchmark_raw.csv", index=False)
    return df


def plot_benchmark(df):
    agg = df.groupby(["dataset", "model"])["acc"].agg(["mean", "std"]).reset_index()
    datasets = list(agg["dataset"].unique())
    models = list(agg["model"].unique())
    fig, ax = plt.subplots(figsize=(10, 4.5))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        sub = agg[agg.model == m].set_index("dataset").loc[datasets]
        ax.bar(np.arange(len(datasets)) + i * w, sub["mean"], w, yerr=sub["std"], capsize=3, label=m)
    ax.set_xticks(np.arange(len(datasets)) + w * (len(models) - 1) / 2)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Test accuracy (mean ± std over seeds)")
    ax.set_ylim(0.4, 1.05)
    ax.set_title("Quantum kernel SVM vs classical baselines")
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_benchmark.png")
    plt.close(fig)
    return agg


# --------------------------------------------------------------------------- #
# 2. Showcase: kernel matrices + decision boundaries on 2-D data
# --------------------------------------------------------------------------- #
def showcase(ds, n_samples, seed=0):
    X, y = qk.load_dataset(ds, n_samples=n_samples, n_features=2, seed=seed)
    Xtr, Xte, ytr, yte = qk.prepare_split(X, y, seed=seed, scale_range=qk.base_range(ds))
    fm = qk.build_feature_map("zz", 2, reps=2)
    K_tr = qk.fidelity_kernel(fm, Xtr)
    K_te = qk.fidelity_kernel(fm, Xte, Xtr)
    clf = qk.PrecomputedKernelSVM(seed=seed).fit(K_tr, ytr)
    acc = accuracy_score(yte, clf.predict(K_te))

    order = np.argsort(ytr)
    ideal = np.outer(2 * ytr[order] - 1, 2 * ytr[order] - 1)
    g = np.linspace(*qk.base_range(ds), 45)
    gx, gy = np.meshgrid(g, g)
    grid = np.c_[gx.ravel(), gy.ravel()]
    Kg = qk.fidelity_kernel(fm, grid, Xtr)
    zz = clf.model_.decision_function(Kg).reshape(gx.shape)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    im = axes[0].imshow(K_tr[np.ix_(order, order)], cmap="viridis", vmin=0, vmax=1)
    axes[0].set_title(f"Quantum kernel (train, sorted by class)\nKTA={qk.kernel_target_alignment(K_tr, ytr):.3f}")
    axes[0].grid(False)
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    axes[1].imshow(ideal, cmap="coolwarm")
    axes[1].set_title("Ideal kernel  y yᵀ")
    axes[1].grid(False)
    axes[2].contourf(gx, gy, zz, levels=20, cmap="RdBu", alpha=0.75)
    axes[2].contour(gx, gy, zz, levels=[0], colors="k", linewidths=1.5)
    axes[2].scatter(*Xtr.T, c=ytr, cmap="bwr", edgecolor="k", s=25, label="train")
    axes[2].scatter(*Xte.T, c=yte, cmap="bwr", edgecolor="k", s=70, marker="*", label="test")
    axes[2].set_title(f"Decision boundary ({ds}), test acc={acc:.3f}")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_showcase_{ds}.png")
    plt.close(fig)
    return acc


def circuit_figure():
    fm = qk.build_feature_map("zz", 3, reps=2)
    fm.decompose().draw("mpl", fold=40, filename=f"{OUT}/fig_feature_map_circuit.png")
    plt.close("all")


# --------------------------------------------------------------------------- #
# 3. Ablations: feature map design (reps / entanglement / data scaling)
# --------------------------------------------------------------------------- #
def ablations(ds, seeds, n_samples):
    rows = []
    n_feat = 2 if ds in ("moons", "circles", "adhoc") else 4
    for seed in seeds:
        X, y = qk.load_dataset(ds, n_samples=n_samples, n_features=n_feat, seed=seed)
        for mult in (0.25, 0.5, 1.0, 2.0):  # bandwidth: features scaled to [0, mult*pi]
            Xtr, Xte, ytr, yte = qk.prepare_split(X, y, seed=seed, scale_range=qk.base_range(ds), scale_mult=mult)
            nq = Xtr.shape[1]
            for fmn, reps, ent in [("zz", 1, "linear"), ("zz", 2, "linear"), ("zz", 3, "linear"),
                                   ("zz", 2, "full"), ("zz", 2, "circular"), ("xyz", 1, "linear"), ("z", 2, "linear")]:
                if nq == 2 and ent == "circular":
                    continue
                fm = qk.build_feature_map(fmn, nq, reps=reps, entanglement=ent)
                r = qk.evaluate_quantum(fm, Xtr, Xte, ytr, yte, seed=seed)
                rows.append({"dataset": ds, "map": fmn, "reps": reps, "entanglement": ent,
                             "scale_mult": mult, "seed": seed, "depth": fm.decompose().depth(), **r})
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/ablation_{ds}.csv", index=False)

    df["config"] = df["map"] + " r" + df["reps"].astype(str) + " " + df["entanglement"]
    piv = df.pivot_table(index="config", columns="scale_mult", values="acc", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    im = ax.imshow(piv.values, cmap="YlGnBu", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{c}×" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_xlabel("bandwidth: features scaled to c × base range")
    ax.set_title(f"Ablation on {ds}: mean test accuracy")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_ablation_{ds}.png")
    plt.close(fig)
    return df


# --------------------------------------------------------------------------- #
# 4. Noise robustness: finite shots + full noisy-circuit simulation
# --------------------------------------------------------------------------- #
def noise_study(ds, n_samples, seeds, shots_list=(32, 128, 512, 2048, 8192)):
    rows = []
    for seed in seeds:
        X, y = qk.load_dataset(ds, n_samples=n_samples, n_features=2, seed=seed)
        Xtr, Xte, ytr, yte = qk.prepare_split(X, y, seed=seed, scale_range=qk.base_range(ds))
        fm = qk.build_feature_map("zz", 2, reps=2)
        Ktr = qk.fidelity_kernel(fm, Xtr)
        Kte = qk.fidelity_kernel(fm, Xte, Xtr)
        rng = np.random.default_rng(seed)
        r = qk.evaluate_quantum(fm, Xtr, Xte, ytr, yte, seed=seed, K_tr=Ktr, K_te=Kte)
        rows.append({"shots": np.inf, "seed": seed, "acc": r["acc"]})
        for s in shots_list:
            Ktr_n = psd_project(qk.shot_noise_kernel(Ktr, s, rng, symmetric=True))
            Kte_n = qk.shot_noise_kernel(Kte, s, rng)
            r = qk.evaluate_quantum(fm, Xtr, Xte, ytr, yte, seed=seed, K_tr=Ktr_n, K_te=Kte_n)
            rows.append({"shots": s, "seed": seed, "acc": r["acc"]})
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/noise_shots_{ds}.csv", index=False)
    agg = df.groupby("shots")["acc"].agg(["mean", "std"]).reset_index()
    finite = agg[np.isfinite(agg.shots)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(finite.shots, finite["mean"], yerr=finite["std"], marker="o", capsize=3, label="finite shots")
    ax.axhline(agg[~np.isfinite(agg.shots)]["mean"].iloc[0], color="g", ls="--", label="exact (∞ shots)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("shots per kernel entry")
    ax.set_ylabel("test accuracy")
    ax.set_title(f"Shot-noise robustness ({ds})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_noise_shots.png")
    plt.close(fig)
    return agg


def hardware_noise_study(ds, seeds, n_train=30, n_test=40):
    """Full Aer simulation with depolarising gate noise (O(n^2) circuits, so small subsets)."""
    rows = []
    for seed in seeds:
        X, y = qk.load_dataset(ds, n_samples=n_train + n_test, n_features=2, seed=seed)
        Xtr, Xte, ytr, yte = qk.prepare_split(X, y, test_size=n_test / (n_train + n_test), seed=seed,
                                              scale_range=qk.base_range(ds))
        fm = qk.build_feature_map("zz", 2, reps=2)
        r = qk.evaluate_quantum(fm, Xtr, Xte, ytr, yte, seed=seed)
        rows.append({"p2q": 0.0, "seed": seed, "acc": r["acc"]})
        for p2 in (0.005, 0.02, 0.05, 0.1):
            Ktr = psd_project(qk.noisy_aer_kernel(fm, Xtr, shots=1024, p1=p2 / 10, p2=p2, seed=seed))
            Kte = qk.noisy_aer_kernel(fm, Xte, Xtr, shots=1024, p1=p2 / 10, p2=p2, seed=seed)
            r = qk.evaluate_quantum(fm, Xtr, Xte, ytr, yte, seed=seed, K_tr=Ktr, K_te=Kte)
            rows.append({"p2q": p2, "seed": seed, "acc": r["acc"]})
        print(f"    seed {seed} done")
    raw = pd.DataFrame(rows)
    raw.to_csv(f"{OUT}/noise_gates_{ds}_raw.csv", index=False)
    df = raw.groupby("p2q")["acc"].agg(["mean", "std"]).reset_index()
    df.to_csv(f"{OUT}/noise_gates_{ds}.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(df.p2q * 100, df["mean"], yerr=df["std"], marker="o", capsize=3, color="C3")
    ax.set_xlabel("two-qubit depolarising error rate (%)")
    ax.set_ylabel("test accuracy")
    ax.set_title(f"Gate-noise robustness, Aer noisy sim ({ds})")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_noise_gates.png")
    plt.close(fig)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    global OUT
    if args.quick:  # smoke-test output goes to its own folder so it never overwrites the full results
        OUT = os.path.join(HERE, "results_quick")
        os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    n = 80 if args.quick else 200
    seeds = [0, 1] if args.quick else [0, 1, 2, 3, 4]
    datasets = ["moons", "adhoc"] if args.quick else ["moons", "circles", "adhoc", "breast_cancer"]

    print("1/5 benchmark");        bench = benchmark(datasets, seeds, n)
    agg = plot_benchmark(bench)
    print("2/5 showcase figures")
    accs = {d: showcase(d, n) for d in ("moons", "adhoc")}
    circuit_figure()
    print("3/5 ablations");        abl = ablations("adhoc", seeds[:3], n)
    print("4/5 shot noise");       shots = noise_study("adhoc", n, seeds)
    print("5/5 gate noise (Aer)")
    gate = hardware_noise_study("adhoc", seeds[:3], n_train=12 if args.quick else 50, n_test=8 if args.quick else 40)

    # ---- summary.md -------------------------------------------------------
    piv = agg.pivot(index="model", columns="dataset", values="mean").round(3)
    sd = agg.pivot(index="model", columns="dataset", values="std").round(3)
    tbl = piv.astype(str) + " ± " + sd.astype(str)
    with open(f"{OUT}/summary.md", "w") as f:
        f.write("# Results summary (auto-generated)\n\n## Test accuracy, mean ± std over seeds\n\n")
        f.write(tbl.to_markdown() + "\n\n")
        f.write("## Shot-noise robustness (adhoc)\n\n" + shots.round(3).to_markdown(index=False) + "\n\n")
        f.write("## Gate-noise robustness (Aer noisy simulation, adhoc, 50 train / 40 test)\n\n" + gate.round(3).to_markdown(index=False) + "\n")
    print(tbl.to_string())
    print(f"\nDone in {time.time() - t0:.0f}s -> {OUT}/")


if __name__ == "__main__":
    main()
