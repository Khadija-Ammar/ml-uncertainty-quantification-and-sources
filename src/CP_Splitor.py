#CP_Splitor.py
# CP_Splitor.py

import numpy as np
import matplotlib.pyplot as plt
from typing import Union


class SplitConformalClassifier:
    """
    Split Conformal Prediction pour la classification binaire.
    Implémente la procédure de Angelopoulos & Bates (2022)
    "A Gentle Introduction to Conformal Prediction".

    Fonctionne indépendamment du modèle sous-jacent tant que
    celui-ci expose une méthode predict_proba(X).

    Parameters
    ----------
    alpha : float
        Niveau d'erreur marginal souhaité (ex: 0.1 pour 90% de couverture).
    """

    def __init__(self, alpha: float = 0.1):
        if not (0 < alpha < 1):
            raise ValueError("alpha doit être dans ]0, 1[.")

        self.alpha        = alpha
        self.q_hat        = None        # quantile calibré
        self.cal_scores   = None        # nonconformity scores sur cal set
        self.is_calibrated = False

    # ------------------------------------------------------------------
    # Nonconformity score
    # ------------------------------------------------------------------

    def _nonconformity_score(self, proba: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calcule le score de non-conformité pour chaque exemple.

        s_i = 1 - p̂(Y_i | X_i)

        Plus le score est élevé, moins la prédiction est confiante
        pour la vraie classe.

        Parameters
        ----------
        proba : np.ndarray, shape (n, 2)
                probabilités prédites pour les classes 0 et 1
        y     : np.ndarray, shape (n,)
                vraies étiquettes (0 ou 1)

        Returns
        -------
        scores : np.ndarray, shape (n,)
        """
        # Probabilité assignée à la VRAIE classe pour chaque exemple
        true_class_proba = proba[np.arange(len(y)), y.astype(int)]
        return 1.0 - true_class_proba

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, model, X_cal: np.ndarray, y_cal: np.ndarray) -> "SplitConformalClassifier":
        """
        Calcule le quantile q̂ sur le calibration set.

        q̂ = quantile ceil((n+1)(1-α)) / n des scores de non-conformité

        Parameters
        ----------
        model : objet avec predict_proba(X) → shape (n, 2)
                (GBUQClassifier, LRUQClassifier ou CalibratedClassifierCV)
        X_cal : features du calibration set
        y_cal : labels du calibration set
        """
        y_cal  = np.array(y_cal)
        proba  = model.predict_proba(X_cal)

        if proba.ndim != 2 or proba.shape[1] != 2:
            raise ValueError("predict_proba doit retourner shape (n, 2).")

        # Scores de non-conformité
        self.cal_scores = self._nonconformity_score(proba, y_cal)

        n = len(self.cal_scores)

        # ✅ Quantile corrigé fini (finite-sample guarantee)
        # q̂ = quantile au niveau ceil((n+1)(1-α))/n
        level    = np.ceil((n + 1) * (1 - self.alpha)) / n
        level    = min(level, 1.0)           # clamp à 1 si alpha très petit
        self.q_hat = np.quantile(self.cal_scores, level, method="higher")

        self.is_calibrated = True

        print(f"[Conformal] n_cal={n} | alpha={self.alpha} | q̂={self.q_hat:.4f}")
        return self

    # ------------------------------------------------------------------
    # Prediction sets
    # ------------------------------------------------------------------

    def predict_set(self, model, X_test: np.ndarray) -> list[list[int]]:
        """
        Retourne les ensembles de prédiction conforme pour chaque exemple.

        C(x) = { y ∈ {0,1} : 1 - p̂(y|x) ≤ q̂ }
             = { y ∈ {0,1} : p̂(y|x) ≥ 1 - q̂ }

        Parameters
        ----------
        model  : objet avec predict_proba(X)
        X_test : features du test set

        Returns
        -------
        prediction_sets : list of lists
            Chaque élément est un sous-ensemble de {0, 1}
            ex : <a href="" class="citation-link" target="_blank" style="vertical-align: super; font-size: 0.8em; margin-left: 3px;">[1]</a>, <a href="" class="citation-link" target="_blank" style="vertical-align: super; font-size: 0.8em; margin-left: 3px;">[0]</a>, [0, 1], []
        """
        if not self.is_calibrated:
            raise ValueError("Appelez calibrate() avant predict_set().")

        proba = model.predict_proba(X_test)      # shape (n, 2)
        threshold = 1.0 - self.q_hat

        prediction_sets = []
        for i in range(len(X_test)):
            pred_set = [c for c in [0, 1] if proba[i, c] >= threshold]
            prediction_sets.append(pred_set)

        return prediction_sets

    # ------------------------------------------------------------------
    # Métriques
    # ------------------------------------------------------------------

    def compute_metrics(self,
                        prediction_sets: list,
                        y_test: np.ndarray) -> dict:
        """
        Calcule les métriques de conformal prediction.

        Parameters
        ----------
        prediction_sets : list of lists, retourné par predict_set()
        y_test          : vraies étiquettes

        Returns
        -------
        dict :
            coverage     : couverture empirique (doit être ≥ 1-alpha)
            avg_set_size : taille moyenne des ensembles
            empty_sets   : proportion d'ensembles vides
            singleton    : proportion d'ensembles singleton {0} ou {1}
            both_classes : proportion d'ensembles {0,1} (incertitude totale)
        """
        y_test = np.array(y_test)
        n      = len(y_test)

        covered     = sum(y_test[i] in prediction_sets[i] for i in range(n))
        sizes       = [len(s) for s in prediction_sets]
        empty       = sum(1 for s in prediction_sets if len(s) == 0)
        singleton   = sum(1 for s in prediction_sets if len(s) == 1)
        both        = sum(1 for s in prediction_sets if len(s) == 2)

        metrics = {
            "coverage"     : covered / n,
            "avg_set_size" : np.mean(sizes),
            "empty_sets"   : empty / n,
            "singleton"    : singleton / n,
            "both_classes" : both / n,
        }

        print("\n=== Conformal Prediction Metrics ===")
        print(f"  Couverture empirique : {metrics['coverage']:.4f}  "
              f"(garantie théorique ≥ {1 - self.alpha:.2f})")
        print(f"  Taille moyenne sets  : {metrics['avg_set_size']:.4f}")
        print(f"  Sets vides           : {metrics['empty_sets']:.4f}")
        print(f"  Singletons           : {metrics['singleton']:.4f}")
        print(f"  Sets {{0,1}}           : {metrics['both_classes']:.4f}")

        return metrics

    # ------------------------------------------------------------------
    # Visualisations
    # ------------------------------------------------------------------

    def plot_nonconformity_scores(self):
        """
        Visualise la distribution des scores de non-conformité
        et la position du quantile q̂.
        """
        if not self.is_calibrated:
            raise ValueError("Appelez calibrate() avant plot_nonconformity_scores().")

        plt.figure(figsize=(8, 5))
        plt.hist(
            self.cal_scores,
            bins=40,
            density=True,
            alpha=0.6,
            color="steelblue",
            edgecolor="black",
            label="Scores de non-conformité"
        )
        plt.axvline(
            self.q_hat,
            color="red",
            linewidth=2,
            linestyle="--",
            label=f"q̂ = {self.q_hat:.4f}  (α={self.alpha})"
        )
        plt.xlabel("Score de non-conformité  s = 1 - p̂(y_true | x)")
        plt.ylabel("Densité")
        plt.title("Distribution des scores de non-conformité (cal set)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    def plot_prediction_sets(self, prediction_sets: list, y_test: np.ndarray):
        """
        Visualise la composition des ensembles de prédiction conforme.

        3 types d'ensembles possibles :
            {0}    → prédit classe 0 avec confiance
            {1}    → prédit classe 1 avec confiance
            {0,1}  → incertain (les deux classes plausibles)
            {}     → ensemble vide (très rare)

        Parameters
        ----------
        prediction_sets : list of lists, retourné par predict_set()
        y_test          : vraies étiquettes (np.ndarray)
        """
        y_test = np.array(y_test)
        n      = len(y_test)

        # ---------------------------------------------------------------
        # Catégorisation des ensembles
        # ---------------------------------------------------------------
        # Convertit chaque ensemble en frozenset pour comparaison facile
        set_types = {
            "vide {}":      0,
            "singleton {0}": 0,
            "singleton {1}": 0,
            "incertain {0,1}": 0,
        }

        for s in prediction_sets:
            fs = frozenset(s)
            if fs == frozenset():
                set_types["vide {}"]           += 1
            elif fs == frozenset({0}):
                set_types["singleton {0}"]     += 1
            elif fs == frozenset({1}):
                set_types["singleton {1}"]     += 1
            elif fs == frozenset({0, 1}):
                set_types["incertain {0,1}"]   += 1

        proportions = {k: v / n for k, v in set_types.items()}

        # ---------------------------------------------------------------
        # Couverture par vraie classe
        # ---------------------------------------------------------------
        n0 = np.sum(y_test == 0)
        n1 = np.sum(y_test == 1)

        # Couverture classe 0 : y_true==0 ET 0 dans l'ensemble prédit
        covered_0 = sum(
            0 in prediction_sets[i]
            for i in range(n) if y_test[i] == 0
        )
        # Couverture classe 1 : y_true==1 ET 1 dans l'ensemble prédit
        covered_1 = sum(
            1 in prediction_sets[i]
            for i in range(n) if y_test[i] == 1
        )
        # Proportion incertains par vraie classe
        uncertain_0 = sum(
            frozenset(prediction_sets[i]) == frozenset({0, 1})
            for i in range(n) if y_test[i] == 0
        )
        uncertain_1 = sum(
            frozenset(prediction_sets[i]) == frozenset({0, 1})
            for i in range(n) if y_test[i] == 1
        )

        cov_labels = [
            "Classe 0\n(no)",
            "Classe 1\n(yes)",
            "Incertains {0,1}\n| vrai=0",
            "Incertains {0,1}\n| vrai=1",
        ]
        cov_values = [
            covered_0   / n0 if n0 > 0 else 0.0,
            covered_1   / n1 if n1 > 0 else 0.0,
            uncertain_0 / n0 if n0 > 0 else 0.0,
            uncertain_1 / n1 if n1 > 0 else 0.0,
        ]

        # ---------------------------------------------------------------
        # Figure
        # ---------------------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # --- Subplot 1 : proportion de chaque type d'ensemble ---
        labels_sets = list(proportions.keys())
        values_sets = list(proportions.values())
        colors_sets = ["gray", "steelblue", "salmon", "orange"]

        bars0 = axes[0].bar(
            labels_sets, values_sets,
            color=colors_sets,
            edgecolor="black"
        )
        axes[0].set_xlabel("Type d'ensemble de prédiction")
        axes[0].set_ylabel("Proportion")
        axes[0].set_title(
            f"Distribution des ensembles de prédiction\n"
            f"(α={self.alpha}, q̂={self.q_hat:.4f})"
        )
        axes[0].set_ylim(0, max(values_sets) * 1.25 if max(values_sets) > 0 else 1)
        axes[0].grid(axis="y", linestyle="--", alpha=0.6)
        for bar, val in zip(bars0, values_sets):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=10
            )

        # --- Subplot 2 : couverture et incertitude par vraie classe ---
        colors_cov = ["steelblue", "salmon", "cornflowerblue", "lightsalmon"]
        bars1 = axes[1].bar(
            cov_labels, cov_values,
            color=colors_cov,
            edgecolor="black"
        )
        axes[1].axhline(
            1 - self.alpha,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Garantie 1-α = {1 - self.alpha:.2f}"
        )
        axes[1].set_ylabel("Proportion")
        axes[1].set_title("Couverture Marginale et incertitude par vraie classe")
        axes[1].set_ylim(0, 1.15)
        axes[1].legend()
        axes[1].grid(axis="y", linestyle="--", alpha=0.6)
        for bar, val in zip(bars1, cov_values):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=10
            )

        plt.tight_layout()
        plt.show()


    def plot_coverage_vs_alpha(self, model,
                                X_cal: np.ndarray, y_cal: np.ndarray,
                                X_test: np.ndarray, y_test: np.ndarray,
                                alphas: np.ndarray = None):
        """
        Trace la couverture empirique en fonction de alpha.
        Permet de vérifier la garantie théorique : couverture ≥ 1 - alpha.

        Parameters
        ----------
        model          : modèle avec predict_proba()
        X_cal, y_cal   : calibration set
        X_test, y_test : test set
        alphas         : niveaux alpha à tester (défaut : 0.01 à 0.5)
        """
        if alphas is None:
            alphas = np.linspace(0.01, 0.5, 50)

        coverages  = []
        set_sizes  = []

        for a in alphas:
            cp = SplitConformalClassifier(alpha=a)
            cp.calibrate(model, X_cal, y_cal)
            pred_sets = cp.predict_set(model, X_test)
            metrics   = cp.compute_metrics(pred_sets, y_test)
            coverages.append(metrics["coverage"])
            set_sizes.append(metrics["avg_set_size"])

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Couverture vs alpha
        axes[0].plot(alphas, coverages,
                     color="steelblue", linewidth=2, label="Couverture empirique")
        axes[0].plot(alphas, 1 - alphas,
                     color="red", linestyle="--",
                     linewidth=1.5, label="Garantie 1 - α")
        axes[0].fill_between(alphas, 1 - alphas, coverages,
                              alpha=0.1, color="green",
                              label="Marge de couverture")
        axes[0].set_xlabel("α")
        axes[0].set_ylabel("Couverture")
        axes[0].set_title("Couverture empirique vs α")
        axes[0].legend()
        axes[0].grid(True, linestyle="--", alpha=0.5)

        # Taille moyenne vs alpha
        axes[1].plot(alphas, set_sizes,
                     color="darkorange", linewidth=2)
        axes[1].set_xlabel("α")
        axes[1].set_ylabel("Taille moyenne des ensembles")
        axes[1].set_title("Taille des ensembles de prédiction vs α")
        axes[1].grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.show()
