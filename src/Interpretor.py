import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt


class UncertaintyInterpreter:
    """
    Analyse des incertitudes produites par SplitConformalClassifier.
    - XGBRegressor pour expliquer l'entropie
    - XGBClassifier pour expliquer la size uncertainty
    """

    def __init__(self):
        self.model_entropy = None
        self.model_size = None
        self.shap_entropy = None
        self.shap_size = None
        self.features = None

    # ----------------------------------------------------------------------
    # Préparation des données
    # ----------------------------------------------------------------------

    def _prepare_data(self, df: pd.DataFrame):
        if "entropy_uncertainty" not in df.columns:
            raise ValueError("Colonne 'entropy_uncertainty' manquante.")

        if "size_uncertainty" not in df.columns:
            raise ValueError("Colonne 'size_uncertainty' manquante.")

        X = df.drop(columns=["entropy_uncertainty", "size_uncertainty"])
        y_entropy = df["entropy_uncertainty"]
        y_size = df["size_uncertainty"] - 1  # Convertir en binaire (0 ou 1)
        
        # split en train_u et test_u pour la modelisation
        

        self.features = X.columns.tolist()
        return X, y_entropy, y_size

    # ----------------------------------------------------------------------
    # Entraînement des modèles
    # ----------------------------------------------------------------------

    def fit_entropy_model(self, df: pd.DataFrame):
        X, y_entropy, _ = self._prepare_data(df)

        self.model_entropy = xgb.XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42
        )

        self.model_entropy.fit(X, y_entropy)
        print("[Interpretor] Modèle XGBRegressor (entropy) entraîné.")

    def fit_size_model(self, df: pd.DataFrame):
        X, _, y_size = self._prepare_data(df)

        self.model_size = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            random_state=42
        )

        self.model_size.fit(X, y_size)
        print("[Interpretor] Modèle XGBClassifier (size) entraîné.")

    # ----------------------------------------------------------------------
    # SHAP VALUES
    # ----------------------------------------------------------------------

    def compute_shap_entropy(self, df: pd.DataFrame):
        if self.model_entropy is None:
            raise ValueError("Modèle entropy non entraîné.")

        X, _, _ = self._prepare_data(df)

        explainer = shap.TreeExplainer(self.model_entropy)
        self.shap_entropy = explainer.shap_values(X)

        print("[Interpretor] SHAP values (entropy) calculées.")
        return self.shap_entropy

    def compute_shap_size(self, df: pd.DataFrame):
        if self.model_size is None:
            raise ValueError("Modèle size non entraîné.")

        X, _, _ = self._prepare_data(df)

        explainer = shap.TreeExplainer(self.model_size)
        self.shap_size = explainer.shap_values(X)

        print("[Interpretor] SHAP values (size) calculées.")
        return self.shap_size

    # ----------------------------------------------------------------------
    # SHAP PLOTS
    # ----------------------------------------------------------------------

    def plot_shap_entropy(self, df: pd.DataFrame):
        if self.shap_entropy is None:
            self.compute_shap_entropy(df)

        X, _, _ = self._prepare_data(df)

        shap.summary_plot(self.shap_entropy, X, show=False)
        plt.title("SHAP Summary Plot — Entropy Uncertainty")
        plt.tight_layout()
        plt.show()

        shap.summary_plot(self.shap_entropy, X, plot_type="bar", show=False)
        plt.title("SHAP Bar Plot — Entropy Uncertainty")
        plt.tight_layout()
        plt.show()

    def plot_shap_size(self, df: pd.DataFrame):
        if self.shap_size is None:
            self.compute_shap_size(df)

        X, _, _ = self._prepare_data(df)

        shap.summary_plot(self.shap_size, X, show=False)
        plt.title("SHAP Summary Plot — Size Uncertainty")
        plt.tight_layout()
        plt.show()

        shap.summary_plot(self.shap_size, X, plot_type="bar", show=False)
        plt.title("SHAP Bar Plot — Size Uncertainty")
        plt.tight_layout()
        plt.show()
