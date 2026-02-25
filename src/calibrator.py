# calibrator.py

import numpy as np
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import (
    roc_curve,
    auc,
    f1_score,
    roc_auc_score,
    classification_report
)

# ==============================================================================
# Calibration
# ==============================================================================

def prob_calibrator(model, X_cal, y_cal, method: str = "sigmoid", n_bins: int = 10):
    """
    Calibre les probabilités d'un modèle déjà fitté et trace les courbes
    de calibration avant/après.

    Parameters
    ----------
    model   : instance de GBUQClassifier ou LRUQClassifier (déjà fitté)
    X_cal   : features de calibration (validation set)
    y_cal   : labels de calibration
    method  : 'sigmoid' ou 'isotonic'
    n_bins  : nombre de bins pour la courbe de calibration

    Returns
    -------
    calibrated_model : CalibratedClassifierCV fitté
    """
    if not model.is_fitted_:
        raise ValueError("Le modèle doit être fitté avant la calibration.")

    # ✅ Correction principale : on passe model.model (sklearn pur) + cv="prefit"
    raw_sklearn_model = model.model

    # ---------- BEFORE calibration ----------
    y_proba = model.predict_proba(X_cal)[:, 1]
    prob_true_before, prob_pred_before = calibration_curve(
        y_cal, y_proba, n_bins=n_bins
    )

    # ---------- Calibration ----------
    calibrated_model = CalibratedClassifierCV(
        estimator=raw_sklearn_model,   # ✅ estimateur sklearn pur
        method=method,
        cv="prefit"                    # ✅ modèle déjà fitté, pas de re-fit
    )
    calibrated_model.fit(X_cal, y_cal)

    # ---------- AFTER calibration ----------
    y_proba_cal = calibrated_model.predict_proba(X_cal)[:, 1]
    prob_true_after, prob_pred_after = calibration_curve(
        y_cal, y_proba_cal, n_bins=n_bins
    )

    # ---------- Plot ----------
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred_before, prob_true_before,
             "o-", label="Avant calibration")
    plt.plot(prob_pred_after, prob_true_after,
             "o-", label="Après calibration")
    plt.plot([0, 1], [0, 1], "k--", label="Calibration parfaite")
    plt.xlabel("Probabilité prédite moyenne")
    plt.ylabel("Fraction de positifs")
    plt.title("Courbes de Calibration")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return calibrated_model


# ==============================================================================
# ECE
# ==============================================================================

def expected_calibration_error(y_true: np.ndarray,
                                y_prob: np.ndarray,
                                n_bins: int = 10) -> float:
    """
    Calcule l'ECE (Expected Calibration Error).

    Parameters
    ----------
    y_true : labels vrais (0/1)
    y_prob : probabilités prédites pour la classe 1
    n_bins : nombre de bins

    Returns
    -------
    ece : float
    """
    # ✅ Fonction standalone, plus de syntaxe @staticmethod hors classe
    bins    = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1

    ece = 0.0
    for i in range(n_bins):
        mask = bin_ids == i
        if np.sum(mask) == 0:
            continue
        bin_acc  = np.mean(y_true[mask])
        bin_conf = np.mean(y_prob[mask])
        ece += (np.sum(mask) / len(y_true)) * abs(bin_acc - bin_conf)

    return ece


def brier_score_loss(y_true, y_prob):
    """
    Calcule le Brier Score Loss.

    Parameters
    ----------
    y_true : labels vrais (0/1)
    y_prob : probabilités prédites pour la classe 1

    Returns
    -------
    brier_score : float
    """
    return np.mean((y_prob - y_true) ** 2)  


# ==============================================================================
# Visualisation ECE + brier score
# ==============================================================================

def plot_ece_comparison(ece_dict: dict):
    """
    Visualise l'ECE de chaque modèle sous forme de barplot.

    Parameters
    ----------
    ece_dict : dict  {label: ece_value}
              ex: {"GB calibré": 0.03, "GB non calibré": 0.08, ...}
    """
    # ✅ Correction : le corps de la fonction était dans la docstring
    labels = list(ece_dict.keys())
    values = list(ece_dict.values())
    colors = ["steelblue" if "calibré" in l else "salmon" for l in labels]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colors, edgecolor="black")
    plt.ylabel("ECE")
    plt.title("Expected Calibration Error (ECE) — Comparaison modèles")
    plt.ylim(0, max(values) * 1.3)

    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=10
        )

    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()
    
def plot_brier_score_comparison(brier_dict: dict):
    """
    Visualise le Brier Score de chaque modèle sous forme de barplot.

    Parameters
    ----------
    brier_dict : dict  {label: brier_value}
                ex: {"GB calibré": 0.15, "GB non calibré": 0.25, ...}
    """
    labels = list(brier_dict.keys())
    values = list(brier_dict.values())
    colors = ["steelblue" if "calibré" in l else "salmon" for l in labels]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colors, edgecolor="black")
    plt.ylabel("Brier Score Loss")
    plt.title("Brier Score Loss — Comparaison modèles")
    plt.ylim(0, max(values) * 1.3)

    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=10
        )

    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

# ==============================================================================
# Evaluation du model calibré sur test set
# ==============================================================================

def evaluate_calibrated_model(calibrated_model, X_test, y_test, label="Modèle calibré"):
    """
    Évalue un modèle calibré sur le test set et affiche AUC, F1, classification report.

    Parameters
    ----------
    calibrated_model : instance de CalibratedClassifierCV
    X_test          : features du test set
    y_test          : labels vrais du test set
    label           : nom du modèle pour l'affichage
    """
    if not hasattr(calibrated_model, "predict_proba"):
        raise ValueError("Le modèle calibré doit avoir une méthode predict_proba.")

    proba  = calibrated_model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= 0.5).astype(int)

    auc_score = roc_auc_score(y_test, proba)
    f1        = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n=== Evaluation {label} ===")
    print(f"  AUC : {auc_score:.4f}")
    print(f"  F1  : {f1:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))