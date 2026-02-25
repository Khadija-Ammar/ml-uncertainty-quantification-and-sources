# model.py

import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_curve,
    auc,
    f1_score,
    roc_auc_score,
    classification_report
)


# ==============================================================================
# Configurations
# ==============================================================================

@dataclass
class GBConfig:
    n_estimators:     int   = 175
    learning_rate:    float = 0.05
    max_depth:        int   = 3
    min_samples_split: int  = 10      # ✅ maintenant passé au modèle
    random_state:     int   = 42
    minority_weight:  float = 8.0     # poids appliqué à la classe minoritaire (y==1)


@dataclass
class LRConfig:
    C:            float = 1.0
    max_iter:     int   = 1000
    random_state: int   = 42
    # ✅ field() retiré, remplacé par None + gestion dans __init__
    class_weight: dict  = None


# ==============================================================================
# Classe de base commune
# ==============================================================================

class BaseUQClassifier:
    """
    Classe de base fournissant predict, predict_proba et evaluate.
    Les sous-classes doivent implémenter fit().
    """

    def __init__(self):
        self.is_fitted_ = False
        self.model      = None

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, X):
        """Retourne les classes prédites."""
        if not self.is_fitted_:
            raise ValueError("Modèle non fitté. Appelez fit() d'abord.")
        return self.model.predict(X)

    def predict_proba(self, X):
        """Retourne les probabilités."""
        if not self.is_fitted_:
            raise ValueError("Modèle non fitté. Appelez fit() d'abord.")
        return self.model.predict_proba(X)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, X, y, threshold: float = 0.5, label: str = ""):
        """
        Évalue le modèle : AUC, F1, classification report.

        Parameters
        ----------
        X         : features
        y         : labels vrais
        threshold : seuil de décision (défaut 0.5)
        label     : nom du modèle pour l'affichage

        Returns
        -------
        dict : {"auc": float, "f1": float}
        """
        if not self.is_fitted_:
            raise ValueError("Modèle non fitté. Appelez fit() d'abord.")

        proba  = self.predict_proba(X)[:, 1]
        y_pred = (proba >= threshold).astype(int)

        auc_score = roc_auc_score(y, proba)
        f1        = f1_score(y, y_pred, zero_division=0)

        # ✅ Correction : suppression de use_calibrated inexistant
        tag = f"[{label}]" if label else ""
        print(f"\n=== Evaluation {tag} ===")
        print(f"  AUC : {auc_score:.4f}")
        print(f"  F1  : {f1:.4f}")
        print(classification_report(y, y_pred, zero_division=0))

        return {"auc": auc_score, "f1": f1}

    # ------------------------------------------------------------------
    # Threshold tuning
    # ------------------------------------------------------------------

    def tune_threshold(self, X_val, y_val, metric: str = "f1"):
        """
        Cherche le meilleur seuil sur le validation set.

        Parameters
        ----------
        X_val, y_val : données de validation
        metric       : métrique à optimiser (seul 'f1' supporté)

        Returns
        -------
        best_threshold : float
        best_score     : float
        """
        if not self.is_fitted_:
            raise ValueError("Modèle non fitté. Appelez fit() d'abord.")

        proba      = self.predict_proba(X_val)[:, 1]
        thresholds = np.linspace(0.01, 0.99, 99)

        best_t, best_score = 0.5, -np.inf

        for t in thresholds:
            y_pred = (proba >= t).astype(int)

            if metric == "f1":
                score = f1_score(y_val, y_pred, zero_division=0)
            else:
                raise ValueError(f"Métrique '{metric}' non supportée. Utilisez 'f1'.")

            if score > best_score:
                best_score = score
                best_t     = t

        print(f"  Meilleur seuil : {best_t:.2f}  |  Meilleur score {metric} : {best_score:.4f}")
        return best_t, best_score


# ==============================================================================
# Gradient Boosting
# ==============================================================================

class GBUQClassifier(BaseUQClassifier):
    """Gradient Boosting avec pénalisation de la classe minoritaire via sample_weight."""

    def __init__(self, config: GBConfig = None):
        super().__init__()
        self.config = config or GBConfig()
        self.model  = GradientBoostingClassifier(
            n_estimators      = self.config.n_estimators,
            learning_rate     = self.config.learning_rate,
            max_depth         = self.config.max_depth,
            min_samples_split = self.config.min_samples_split,  # ✅ ajouté
            random_state      = self.config.random_state
        )

    def fit(self, X_train, y_train):
        """
        Entraîne le GB avec sample_weight calculé dynamiquement
        pour pénaliser la classe minoritaire (y==1).
        """
        sample_weight = np.where(
            np.array(y_train) == 1,
            self.config.minority_weight,
            1.0
        )
        self.model.fit(X_train, y_train, sample_weight=sample_weight)
        self.is_fitted_ = True
        return self


# ==============================================================================
# Logistic Regression
# ==============================================================================

class LRUQClassifier(BaseUQClassifier):
    """Régression Logistique avec pénalisation via class_weight."""

    def __init__(self, config: LRConfig = None):
        super().__init__()
        self.config = config or LRConfig()

        # ✅ Gestion du default mutable class_weight
        class_weight = self.config.class_weight if self.config.class_weight is not None \
                       else {0: 1.0, 1: 8.0}

        self.model = LogisticRegression(
            C            = self.config.C,
            max_iter     = self.config.max_iter,
            random_state = self.config.random_state,
            class_weight = class_weight
        )

    def fit(self, X_train, y_train):
        """Entraîne la régression logistique."""
        self.model.fit(X_train, y_train)
        self.is_fitted_ = True
        return self


# ==============================================================================
# Courbe ROC AUC
# ==============================================================================

def plot_roc_curves(classifiers: dict, X, y):
    """
    Trace les courbes ROC pour plusieurs modèles.

    Parameters
    ----------
    classifiers : dict  {label: classifier}
    X           : features
    y           : labels vrais
    """
    plt.figure(figsize=(7, 6))
    for label, clf in classifiers.items():
        proba = clf.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, proba)
        roc_auc     = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Courbes ROC")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


