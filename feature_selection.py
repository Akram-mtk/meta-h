"""
Feature Selection with Metaheuristics
TP N°5 / N°6 / N°7 / N°8 - MÉTA - Master 2 SII - USTHB
"""
import numpy as np
from sklearn.datasets import load_digits, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# =============================================================================
# Dataset loaders (cached by the Streamlit layer via st.cache_data)
# =============================================================================
def load_dataset(name: str):
    """Return (X, y, n_features) for the requested dataset."""
    if name == "Digits":
        data = load_digits()
        X, y = data.data, data.target
    elif name == "Synthetic":
        X, y = make_classification(
            n_samples=1000,
            n_features=50,
            n_informative=5,
            n_redundant=10,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return X, y, X.shape[1]


# =============================================================================
# Fitness evaluator (closure pattern — returns a callable ready for PSO/GA)
# =============================================================================
class FSEvaluator:
    """
    Fitness evaluator for feature selection.

    Supports two encoding modes:
      - "continuous": x_i in [0, 1]. Top-k or threshold-based selection.
                      Used by PSO (TP5/6).
      - "binary":     x_i in {0, 1}. Direct selection mask.
                      Used by GA (TP7/8).

    The objective function (from TP5):
        f(x) = alpha * E_R(x) + (1 - alpha) * F_R(x)
        E_R  = 1 - accuracy (classification error of KNN)
        F_R  = N_s / N        (fraction of selected features)
    """

    def __init__(self, X, y, alpha=0.99, n_selected=None,
                 k_neighbors=5, test_size=0.3, random_state=42,
                 mode="continuous"):
        self.X = X
        self.y = y
        self.alpha = float(alpha)
        self.n_selected = n_selected    # fixed number of features (None = threshold)
        self.mode = mode
        self.n_features = X.shape[1]

        # Train/test split — fixed so fitness is comparable across runs
        self.X_tr, self.X_te, self.y_tr, self.y_te = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        self.knn = KNeighborsClassifier(n_neighbors=k_neighbors)

    # ------------------------------------------------------------------
    def select_indices(self, x: np.ndarray) -> np.ndarray:
        """Convert a solution x into the list of selected feature indices."""
        x = np.asarray(x)
        if self.mode == "binary":
            return np.where(x > 0.5)[0]
        # continuous mode
        if self.n_selected is None:
            # threshold: keep features where x_i > 0.5
            return np.where(x > 0.5)[0]
        # top-k
        k = min(self.n_selected, self.n_features)
        return np.argsort(-x)[:k]

    # ------------------------------------------------------------------
    def evaluate(self, x: np.ndarray):
        """
        Return (fitness, accuracy, n_selected).
        Fitness is to be minimized.
        """
        idx = self.select_indices(x)
        if len(idx) == 0:
            # No feature selected → worst possible fitness
            return 1.0, 0.0, 0
        Xtr = self.X_tr[:, idx]
        Xte = self.X_te[:, idx]
        self.knn.fit(Xtr, self.y_tr)
        y_pred = self.knn.predict(Xte)
        acc = accuracy_score(self.y_te, y_pred)
        err = 1.0 - acc
        fr = len(idx) / self.n_features
        f = self.alpha * err + (1.0 - self.alpha) * fr
        return float(f), float(acc), int(len(idx))

    # ------------------------------------------------------------------
    def __call__(self, x: np.ndarray) -> float:
        """Fitness-only — signature compatible with PSO/GA drivers."""
        f, _, _ = self.evaluate(x)
        return f
