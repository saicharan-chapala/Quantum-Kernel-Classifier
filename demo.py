"""
demo.py - 30-second live demo: train a quantum kernel classifier and compare to classical RBF.

    python demo.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # lets VS Code run this from any working directory

import numpy as np
from sklearn.metrics import accuracy_score

import qkernel as qk

# 1) Data: quantum-native 'ad hoc' set (Havlicek et al. style), 2 features -> 2 qubits
X, y = qk.load_dataset("adhoc", n_samples=200, seed=0)
Xtr, Xte, ytr, yte = qk.prepare_split(X, y, seed=0, scale_range=qk.base_range("adhoc"))

# 2) Feature map: ZZ feature map, 2 reps, linear entanglement
fmap = qk.build_feature_map("zz", n_qubits=2, reps=2, entanglement="linear")
print(fmap.decompose().draw(fold=100))

# 3) Quantum kernel (Gram matrices) -> SVM with precomputed kernel
K_train = qk.fidelity_kernel(fmap, Xtr)          # (n_train, n_train)
K_test = qk.fidelity_kernel(fmap, Xte, Xtr)      # (n_test,  n_train)
clf = qk.PrecomputedKernelSVM(seed=0).fit(K_train, ytr)
acc_q = accuracy_score(yte, clf.predict(K_test))

# 4) Classical baseline (tuned RBF SVM)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler().fit(Xtr)
acc_c = qk.evaluate_classical("rbf", sc.transform(Xtr), sc.transform(Xte), ytr, yte)["acc"]

print(f"\nQuantum-kernel SVM  test accuracy: {acc_q:.3f}   (C={clf.C_}, KTA={qk.kernel_target_alignment(K_train, ytr):.3f})")
print(f"Classical RBF SVM   test accuracy: {acc_c:.3f}")
