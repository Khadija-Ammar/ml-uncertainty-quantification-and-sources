# Interpretor.py

import numpy as np
import pandas as pd
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt


class UncertaintyInterpreter:
    """
    Analyse des incertitudes produites par SplitConformalClassifier.
    Entraîne deux modèles LightGBM :
        - un pour expliquer l'entropie
        - un pour expliquer la height uncertainty
    Et calcule les valeurs de Shapley pour identifier les features
    qui génèrent le plus d'incertitude.
    """

    def __init__(self):
        self.model_entropy = None
        self.model_height = None
        self.shap_entropy = None
        self.shap_height = None
        self.features = None

    # ----------------------------------------------------------------------
    # Préparation des données
    # ----------------------------------------------------------------------

    def _prepare_data(self, df: pd.DataFrame):
        """
        Sépare les features et les deux cibles d'incertitude.
        """
        if "entropy_uncertainty" not in df.columns:
            raise ValueError("Colonne 'entropy_uncertainty' manquante dans df.")

        if "height_uncertainty" not in df.columns:
            raise ValueError("Colonne 'height_uncertainty' manquante dans df.")

        X = df.drop(columns=["entropy_uncertainty", "height_uncertainty"])
        y_entropy = df["entropy_uncertainty"]
        y_height = df["height_uncertainty"]

        self.features = X.columns.tolist()

        return X, y_entropy, y_height

    # ----------------------------------------------------------------------
    # Entraînement des modèles LightGBM
    # ----------------------------------------------------------------------

    def fit_entropy_model(self, df: pd.DataFrame):
        """
        Entraîne un modèle LightGBM pour prédire l'entropie.
        """
        X, y_entropy, _ = self._prepare_data(df)

        self.model_entropy = lgb.LGBMRegressor(
            n_estimators=400,
            max_depth=-1,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        self.model_entropy.fit(X, y_entropy)
        print("[Interpretor] Modèle LightGBM (entropy) entraîné.")

    def fit_height_model(self, df: pd.DataFrame):
        """
        Entraîne un modèle LightGBM pour prédire la height uncertainty.
        """
        X, _, y_height = self._prepare_data(df)

        self.model_height = lgb.LGBMRegressor(
            n_estimators=400,
            max_depth=-1,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        self.model_height.fit(X, y_height)
        print("[Interpretor] Modèle LightGBM (height) entraîné.")

    # ----------------------------------------------------------------------
    # SHAP VALUES
    # ----------------------------------------------------------------------

    def compute_shap_entropy(self, df: pd.DataFrame):
        """
        Calcule les valeurs SHAP pour le modèle d'entropie.
        """
        if self.model_entropy is None:
            raise ValueError("Modèle entropy non entraîné.")

        X, _, _ = self._prepare_data(df)

        explainer = shap.TreeExplainer(self.model_entropy)
        self.shap_entropy = explainer.shap_values(X)

        print("[Interpretor] SHAP values (entropy) calculées.")
        return self.shap_entropy

    def compute_shap_height(self, df: pd.DataFrame):
        """
        Calcule les valeurs SHAP pour le modèle de height uncertainty.
        """
        if self.model_height is None:
            raise ValueError("Modèle height non entraîné.")

        X, _, _ = self._prepare_data(df)

        explainer = shap.TreeExplainer(self.model_height)
        self.shap_height = explainer.shap_values(X)

        print("[Interpretor] SHAP values (height) calculées.")
        return self.shap_height

    # ----------------------------------------------------------------------
    # PLOTS
    # ----------------------------------------------------------------------

    def plot_shap_entropy(self, df: pd.DataFrame):
        """
        Summary plot SHAP pour l'entropie.
        """
        if self.shap_entropy is None:
            self.compute_shap_entropy(df)

        X, _, _ = self._prepare_data(df)

        shap.summary_plot(self.shap_entropy, X, show=False)
        plt.title("SHAP Summary Plot — Entropy Uncertainty")
        plt.tight_layout()
        plt.show()

    def plot_shap_height(self, df: pd.DataFrame):
        """
        Summary plot SHAP pour la height uncertainty.
        """
        if self.shap_height is None:
            self.compute_shap_height(df)

        X, _, _ = self._prepare_data(df)

        shap.summary_plot(self.shap_height, X, show=False)
        plt.title("SHAP Summary Plot — Height Uncertainty")
        plt.tight_layout()
        plt.show()

    # ----------------------------------------------------------------------
    # TOP FEATURES
    # ----------------------------------------------------------------------

    def get_top_features(self, df: pd.DataFrame, k=10):
        """
        Retourne les k features qui contribuent le plus à l'incertitude.
        """
        if self.shap_entropy is None:
            self.compute_shap_entropy(df)

        if self.shap_height is None:
            self.compute_shap_height(df)

        X, _, _ = self._prepare_data(df)

        mean_abs_entropy = np.abs(self.shap_entropy).mean(axis=0)
        mean_abs_height = np.abs(self.shap_height).mean(axis=0)

        df_entropy = pd.DataFrame({
            "feature": self.features,
            "importance_entropy": mean_abs_entropy
        }).sort_values("importance_entropy", ascending=False)

        df_height = pd.DataFrame({
            "feature": self.features,
            "importance_height": mean_abs_height
        }).sort_values("importance_height", ascending=False)

        return df_entropy.head(k), df_height.head(k)
