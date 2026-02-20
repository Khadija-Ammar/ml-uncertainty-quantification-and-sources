# inference_prop_pred.py

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# ==============================================================================
# Distribution des probabilités prédites
# ==============================================================================

def plot_proba_distributions(classifiers: dict, X, title_suffix: str = ""):
    """
    Visualise la distribution des probabilités prédites (histogramme + KDE).

    Parameters
    ----------
    classifiers  : dict {label: classifier}
                   classifier peut être un GBUQClassifier, LRUQClassifier
                   ou un CalibratedClassifierCV
    X            : features
    title_suffix : str affiché en bas du titre de chaque subplot
    """
    n = len(classifiers)
    if n == 0:
        raise ValueError("Le dictionnaire de classifiers est vide.")

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=False)

    # ✅ Correction : axes toujours une liste même si n==1
    if n == 1:
        axes = [axes]

    for ax, (label, clf) in zip(axes, classifiers.items()):

        # ✅ Gestion des deux types : wrapper custom ou sklearn pur
        proba = clf.predict_proba(X)[:, 1]

        # Histogramme
        ax.hist(
            proba,
            bins=40,
            density=True,
            alpha=0.6,
            color="steelblue",
            edgecolor="black",
            label="Histogramme"
        )

        # KDE avec scipy
        kde_x = np.linspace(0, 1, 300)
        kde   = stats.gaussian_kde(proba)
        ax.plot(kde_x, kde(kde_x), color="red", linewidth=2, label="KDE")

        # Ligne médiane
        ax.axvline(
            np.median(proba),
            color="orange",
            linestyle="--",
            linewidth=1.5,
            label=f"Médiane : {np.median(proba):.2f}"
        )

        ax.set_xlabel("Probabilité prédite")
        ax.set_ylabel("Densité")
        ax.set_title(f"Distribution des probas\n{label} {title_suffix}")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


# ==============================================================================
# Test de Kolmogorov-Smirnov
# ==============================================================================

def ks_test_analysis(classifiers: dict, X) -> dict:
    """
    Effectue le test de Kolmogorov-Smirnov entre les distributions
    de probabilités prédites des modèles deux à deux.

    Parameters
    ----------
    classifiers : dict {label: classifier}
    X           : features

    Returns
    -------
    dict : {(label1, label2): {"statistic": float, "p_value": float}}
    """
    if len(classifiers) < 2:
        raise ValueError("Il faut au moins 2 classifiers pour le test KS.")

    labels = list(classifiers.keys())
    probas = {
        label: clf.predict_proba(X)[:, 1]
        for label, clf in classifiers.items()
    }

    results = {}
    print("\n=== Test de Kolmogorov-Smirnov entre distributions ===")

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            l1, l2       = labels[i], labels[j]
            ks_stat, p_value = stats.ks_2samp(probas[l1], probas[l2])
            results[(l1, l2)] = {"statistic": ks_stat, "p_value": p_value}

            print(f"\n  {l1}  vs  {l2}")
            print(f"    KS statistic : {ks_stat:.4f}")
            print(f"    p-value      : {p_value:.4e}")
            if p_value < 0.05:
                print("    → Distributions SIGNIFICATIVEMENT différentes (α=0.05)")
            else:
                print("    → Pas de différence significative (α=0.05)")

    return results



