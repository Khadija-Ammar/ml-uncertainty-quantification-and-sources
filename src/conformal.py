import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency

# =============================================================================
# 1) Quantile (garantie en échantillon fini)
# =============================================================================

def quantile(scores, alpha: float) -> float:
    scores = np.asarray(scores, dtype=float)
    n = scores.shape[0]
    if n == 0:
        raise ValueError("scores est vide : impossible de calibrer un quantile conformal.")

    # k = ceil((n+1)*(1-alpha)) : on veut accepter les 90% "meilleures" erreurs.
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return float(np.sort(scores)[k - 1])

# =============================================================================
# 2) Fit Split CP : apprendre q_hat et tau sur le CALIBRATION set
# =============================================================================
def fit_split_cp_proba(clf, X_cal, y_cal, alpha: float = 0.1) -> dict:
    """
    Calibre Split Conformal à partir d'un modèle probabiliste.

    Étapes :
    1) On calcule p_hat(y_i | x_i) pour chaque point du CALIBRATION set.
    2) On définit le score au vrai label :
           s_i = 1 - p_hat(y_i | x_i)
    3) On prend q_hat = quantile_(1-alpha) des scores.
    4) On définit tau = 1 - q_hat (seuil proba équivalent).

    Interprétation de tau :
    - Un label y sera inclus dans Gamma(x) si p_hat(y|x) >= tau.
    - En binaire, si tau est haut, c'est "strict" => plus de sets {0,1} (incertains).
      Si tau est bas, c'est "laxiste" => plus de sets de taille 1.
    """
    # predict_proba retourne (n, K) avec K=2 pour binaire
    proba = clf.predict_proba(X_cal)
    y_cal = np.asarray(y_cal, dtype=int)

    if proba.ndim != 2:
        raise ValueError("predict_proba doit retourner un array 2D (n_samples, n_classes).")

    n, K = proba.shape
    if y_cal.shape[0] != n:
        raise ValueError("X_cal et y_cal n'ont pas le même nombre d'observations.")

    # Probabilité du vrai label pour chaque exemple
    p_true = proba[np.arange(n), y_cal]

    # Nonconformity score au vrai label
    scores = 1.0 - p_true

    # Quantile conformal
    qhat = quantile(scores, alpha)

    # Seuil proba équivalent
    tau = 1.0 - qhat

    return {
        "alpha": float(alpha),
        "qhat": float(qhat),
        "tau": float(tau),
        "cal_scores": scores,
    }


# =============================================================================
# 3) Construire les prediction sets Gamma(x) à partir de tau
# =============================================================================

def predict_sets_from_tau(clf, X, tau: float) -> list[list[int]]:
    """
    Construit les prediction sets Gamma(x) pour chaque x.

    Règle d'inclusion :
        inclure le label y si p_hat(y|x) >= tau

    - En binaire, Gamma(x) peut être {0}, {1} ou {0,1}.
    - En multiclass, Gamma(x) peut contenir plusieurs labels.

    Safeguard :
    - Il peut arriver (rarement) qu'aucun label ne passe le seuil tau
      (ex: tau très grand). Pour éviter un set vide, on met alors argmax(proba).
    """
    proba = clf.predict_proba(X)
    if proba.ndim != 2:
        raise ValueError("predict_proba doit retourner un array 2D (n_samples, n_classes).")

    pred_sets = []
    for p in proba:
        # labels inclus
        s = [int(y) for y, py in enumerate(p) if py >= tau]

        # éviter set vide
        if len(s) == 0:
            s = [int(np.argmax(p))]

        pred_sets.append(s)

    return pred_sets


# =============================================================================
# 4) Métriques CP : coverage & efficiency
# =============================================================================

def coverage_efficiency(pred_sets, y_true) -> dict:
    """
    Calcule les métriques classiques en Conformal Prediction sur un dataset.

    - Coverage marginale :
        fraction des i tels que y_i appartient à Gamma(x_i)
    - Efficiency (classification) :
        taille moyenne des prediction sets |Gamma(x)|
    """
    y_true = np.asarray(y_true, dtype=int)
    n = y_true.shape[0]
    if len(pred_sets) != n:
        raise ValueError("pred_sets et y_true doivent avoir la même longueur.")

    covered = np.array([y_true[i] in pred_sets[i] for i in range(n)], dtype=float)
    sizes = np.array([len(s) for s in pred_sets], dtype=float)

    return {
        "coverage": float(covered.mean()),
        "avg_set_size": float(sizes.mean()),
        "sizes": sizes,
        "covered": covered,
    }


# =============================================================================
# 5) Définir U(x) à partir des outputs conformal
# =============================================================================

def compute_U(clf, X, tau: float) -> dict:
    """
    Calcule des variables d'incertitude U(x) dérivées de Conformal Prediction.

    U (discret, standard CP) :
        U(x) = |Gamma(x)|
        - en binaire : 1 (tranche) ou 2 (incertain)

    """
    proba = clf.predict_proba(X)
    if proba.ndim != 2:
        raise ValueError("predict_proba doit retourner un array 2D (n_samples, n_classes).")

    pmax = np.max(proba, axis=1)
    pred_sets = predict_sets_from_tau(clf, X, tau=tau)

    U = np.array([len(s) for s in pred_sets], dtype=int)

    return {
        "U": U,
        "pred_sets": pred_sets,
        "pmax": pmax,
        "proba": proba,
    }

# =============================================================================
# 6) Distribution globale de U
# =============================================================================
def distribution(clf,X,U):
    df=pd.DataFrame({"U":U, "p1":clf.predict_proba(X)[:,1]})
    print(df["U"].describe())
    plt.figure(figsize=(6,4))
    sns.countplot(x="U", data=df)
    plt.title("Distribution de U = |Gamma(x)|")
    plt.xlabel("Taille du prediction set")
    plt.ylabel("Nombre d'observations")
    plt.tight_layout()
    plt.show()
    print("\nProportion incertaine (U=2):", (df["U"]==2).mean())
# =============================================================================
# 7) tests d'hypothèses
# =============================================================================
"""
Hypothèse : Incertitude et proximité de la frontière
H0 : E[∣p−0.5∣∣U=1]=E[∣p−0.5∣∣U=2]
U(x)=2 plus fréquent quand p1(x) est proche de 0.5.
Test : comparer abs(p1-0.5) entre U=1 et U=2.
"""
def test1(clf,X,U):
    df=pd.DataFrame({"U":U, "p1":clf.predict_proba(X)[:,1]})
    df["margin"] = np.abs(df["p1"] - 0.5)

    m1 = df[df["U"]==1]["margin"]
    m2 = df[df["U"]==2]["margin"]

    t_stat, p_val = ttest_ind(m1, m2, equal_var=False)

    print("\n=== TEST H1: Margin vs U ===")
    print("Mean margin U=1:", m1.mean())
    print("Mean margin U=2:", m2.mean())
    print("t-stat:", t_stat)
    print("p-value:", p_val)

"""
H2 — Différences par sous-groupes 
Hypothèse : certains groupes ont une moyenne de U différente.
Test : tableau groupby + chi2 sur “incertain (U=2)”.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency, ttest_ind

from typing import Optional

def test2(
    df_test_raw: pd.DataFrame,
    U: np.ndarray,
    p1: Optional[np.ndarray] = None,
    y_true: Optional[np.ndarray] = None,
    group_cols: Optional[list] = None, # <--- Corrigé ici
    age_col: str = "age",
    top_k_groups: int = 15
) -> dict:
    """
    Stats descriptives + tests sur U.

    Parameters
    ----------
    df_test_raw : DataFrame (features originales, non transformées) du test set
    U           : array (n_test,) avec U = |Gamma(x)| (binaire: 1 ou 2)
    p1          : array (n_test,) avec P(Y=1|X) (optionnel mais très utile)
    y_true      : labels vrais (optionnel)
    group_cols  : colonnes catégorielles à tester (job, contact, poutcome, marital, etc.)
    age_col     : nom de la colonne âge
    top_k_groups: pour les plots, limite du nombre de catégories affichées

    Returns
    -------
    results : dict contenant tables + p-values
    """

    # ---------- checks ----------
    U = np.asarray(U, dtype=int)
    if len(df_test_raw) != len(U):
        raise ValueError(f"df_test_raw ({len(df_test_raw)}) et U ({len(U)}) doivent avoir même longueur.")

    df = df_test_raw.copy()
    df["U"] = U
    df["uncertain"] = (df["U"] >= 2).astype(int)  # binaire: 1 si set {0,1}

    results = {}

    # ============================================================
    # A) Distribution globale de U
    # ============================================================
    counts = df["U"].value_counts().sort_index()
    props = counts / len(df)

    print("\n=== Distribution de U ===")
    print(counts)
    print("\nProportions:")
    print(props)

    plt.figure(figsize=(6,4))
    counts.plot(kind="bar")
    plt.title("Distribution de U = |Gamma(x)|")
    plt.xlabel("Taille du prediction set")
    plt.ylabel("Nombre d'observations")
    plt.tight_layout()
    plt.show()

    results["counts_U"] = counts
    results["props_U"] = props

    # ============================================================
    # B) U vs marge (si p1 fourni)
    # ============================================================
    if p1 is not None:
        p1 = np.asarray(p1, dtype=float)
        if len(p1) != len(df):
            raise ValueError("p1 doit avoir la même longueur que df_test_raw.")

        df["p1"] = p1
        df["margin"] = np.abs(df["p1"] - 0.5)

        m1 = df.loc[df["U"] == 1, "margin"]
        m2 = df.loc[df["U"] >= 2, "margin"]

        t_stat, p_val = ttest_ind(m1, m2, equal_var=False)

        print("\n=== Test H1 : margin vs U ===")
        print(f"Mean margin (U=1): {m1.mean():.4f}")
        print(f"Mean margin (U=2): {m2.mean():.4f}")
        print(f"t-stat = {t_stat:.4f} | p-value = {p_val:.3e}")

        plt.figure(figsize=(6,4))
        plt.boxplot([m1, m2], labels=["U=1", "U=2"])
        plt.title("Distance à la frontière |p1-0.5| selon U")
        plt.ylabel("|p1 - 0.5|")
        plt.tight_layout()
        plt.show()

        results["H1_margin_ttest_pvalue"] = float(p_val)

    # ============================================================
    # C) U vs erreur (si y_true fourni)
    # ============================================================
    if y_true is not None and p1 is not None:
        y_true = np.asarray(y_true, dtype=int)
        if len(y_true) != len(df):
            raise ValueError("y_true doit avoir la même longueur que df_test_raw.")

        # prédiction classique (threshold 0.5)
        y_pred = (df["p1"].values >= 0.5).astype(int)
        df["error"] = (y_pred != y_true).astype(int)

        u_err = df.loc[df["error"] == 1, "U"]
        u_ok  = df.loc[df["error"] == 0, "U"]

        t_stat, p_val = ttest_ind(u_err, u_ok, equal_var=False)

        print("\n=== Test H2 : U vs erreur ===")
        print(f"Mean U (correct): {u_ok.mean():.4f}")
        print(f"Mean U (error)  : {u_err.mean():.4f}")
        print(f"t-stat = {t_stat:.4f} | p-value = {p_val:.3e}")

        results["H2_U_error_ttest_pvalue"] = float(p_val)

    # ============================================================
    # D) Group comparisons (chi2) : uncertain vs group
    # ============================================================
    if group_cols is None:
        # choix par défaut (top features utiles)
        group_cols = ["job", "contact", "poutcome", "marital", "education", "housing", "loan"]

    group_cols = [c for c in group_cols if c in df.columns]

    for col in group_cols:
        # table de contingence
        tab = pd.crosstab(df[col], df["uncertain"])
        if tab.shape[0] < 2:
            continue

        chi2, p, dof, exp = chi2_contingency(tab)

        print(f"\n=== Chi2 : uncertain vs {col} ===")
        print(f"p-value = {p:.3e}")

        # plot top_k_groups catégories (par proportion d'incertain)
        rate = df.groupby(col)["uncertain"].mean().sort_values(ascending=False)
        rate_plot = rate.head(top_k_groups)

        plt.figure(figsize=(8,5))
        rate_plot.sort_values().plot(kind="barh")
        plt.title(f"P(U=2 | {col}) — top {top_k_groups}")
        plt.xlabel("Proportion incertaine (U=2)")
        plt.tight_layout()
        plt.show()

        results[f"chi2_pvalue_{col}"] = float(p)
        results[f"rate_uncertain_{col}"] = rate

    # ============================================================
    # E) Age analysis : bins + plot
    # ============================================================
    if age_col in df.columns:
        df["age_bin"] = pd.cut(df[age_col], bins=[0,30,40,50,60,100], right=True)
        age_rate = df.groupby("age_bin")["uncertain"].mean()

        print("\n=== P(U=2 | age_bin) ===")
        print(age_rate)

        plt.figure(figsize=(8,4))
        age_rate.plot(kind="bar")
        plt.title("P(U=2 | Age bins)")
        plt.ylabel("Proportion incertaine")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        results["rate_uncertain_age_bin"] = age_rate

    return results
def eda_u_distribution(U: np.ndarray) -> pd.DataFrame:
    """
    EDA simple de la variable discrète d'incertitude U.

    Parameters
    ----------
    U : array-like
        Taille du prediction set |Gamma(x)|.

    Returns
    -------
    summary : DataFrame
        Tableau avec effectifs et proportions.
    """
    U = np.asarray(U, dtype=int)

    counts = pd.Series(U).value_counts().sort_index()
    props = counts / len(U)

    summary = pd.DataFrame({
        "count": counts,
        "proportion": props
    })

    print("\n=== EDA: distribution de U ===")
    print(summary)

    plt.figure(figsize=(6, 4))
    sns.countplot(x=U)
    plt.title("Distribution de U = |Gamma(x)|")
    plt.xlabel("Valeur de U")
    plt.ylabel("Nombre d'observations")
    plt.tight_layout()
    plt.show()

    return summary


def eda_u_vs_proba(clf, X, U: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """
    Analyse la relation entre U et la probabilité prédite p1 = P(Y=1|X).

    Parameters
    ----------
    clf : modèle avec predict_proba
    X   : features
    U   : array-like
    bins: int
        Nombre de classes pour discrétiser p1.

    Returns
    -------
    bin_summary : DataFrame
        Tableau par bins de proba avec taux d'incertitude.
    """
    U = np.asarray(U, dtype=int)
    p1 = clf.predict_proba(X)[:, 1]

    df = pd.DataFrame({
        "p1": p1,
        "U": U,
        "uncertain": (U >= 2).astype(int),
        "margin": np.abs(p1 - 0.5)
    })

    print("\n=== EDA: U vs probabilité prédite ===")
    print(df[["p1", "margin"]].describe())

    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x="p1", hue="U", bins=30, stat="density", common_norm=False)
    plt.axvline(0.5, linestyle="--")
    plt.title("Distribution de p(y=1|x) selon U")
    plt.xlabel("Probabilité prédite p1")
    plt.tight_layout()
    plt.show()

    df["p1_bin"] = pd.cut(df["p1"], bins=bins)
    bin_summary = df.groupby("p1_bin", observed=False)["uncertain"].agg(["mean", "count"])
    bin_summary = bin_summary.rename(columns={"mean": "uncertainty_rate"})

    print("\n=== Taux d'incertitude par bin de p1 ===")
    print(bin_summary)

    plt.figure(figsize=(8, 4))
    plt.plot(range(len(bin_summary)), bin_summary["uncertainty_rate"], marker="o")
    plt.title("Taux d'incertitude selon les bins de p1")
    plt.xlabel("Bin de probabilité")
    plt.ylabel("P(U=2)")
    plt.xticks(range(len(bin_summary)), [str(i) for i in bin_summary.index], rotation=45)
    plt.tight_layout()
    plt.show()

    return bin_summary


def eda_u_by_group(
    df_test_raw: pd.DataFrame,
    U: np.ndarray,
    group_cols: list[str],
    top_k: int = 10
) -> dict:
    """
    Analyse descriptive du taux d'incertitude par groupe.

    Parameters
    ----------
    df_test_raw : DataFrame
        Données test brutes (non encodées de préférence).
    U : array-like
        Variable d'incertitude.
    group_cols : list[str]
        Colonnes catégorielles à analyser.
    top_k : int
        Nombre maximum de modalités affichées.

    Returns
    -------
    results : dict
        Dictionnaire de séries pandas avec taux d'incertitude par groupe.
    """
    U = np.asarray(U, dtype=int)

    df = df_test_raw.copy()
    df["U"] = U
    df["uncertain"] = (df["U"] >= 2).astype(int)

    results = {}

    for col in group_cols:
        if col not in df.columns:
            continue

        rate = df.groupby(col)["uncertain"].mean().sort_values(ascending=False)
        results[col] = rate

        print(f"\n=== EDA: taux d'incertitude par {col} ===")
        print(rate.head(top_k))

        plt.figure(figsize=(8, 5))
        rate.head(top_k).sort_values().plot(kind="barh")
        plt.title(f"Taux d'incertitude P(U=2) par {col}")
        plt.xlabel("Proportion incertaine")
        plt.tight_layout()
        plt.show()

    return results

