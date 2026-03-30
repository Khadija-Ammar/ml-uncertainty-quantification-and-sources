# preprocessor.py

import pandas as pd
import numpy as np
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline


@dataclass
class SplitData:
    """Container for train/val/test splits."""
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


class Preprocessor:
    """
    Preprocessor class that:
    - detecte types de features
    - construit ColumnTransformer (scaling + onehot)
    - fit/transform et retourne DataFrames avec noms de colonnes
    """

    def __init__(
        self,
        target_column: str = "deposit",
        drop_columns: list = None,
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 42
    ):
        self.target_column = target_column
        self.drop_columns = drop_columns if drop_columns is not None else []
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        self.preprocessor = None
        self.numeric_features = None
        self.categorical_features = None
        self.feature_names_ = None   # noms finaux après fit
        self.label_encoder = LabelEncoder()

    def drop_unwanted_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns that should not be used."""
        return df.drop(columns=self.drop_columns, errors="ignore")

    def encode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode la variable cible en valeurs numériques (0/1).
        """
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset.")

        df = df.copy()
        df[self.target_column] = self.label_encoder.fit_transform(df[self.target_column])

        print(f"[encode_target] Classes encodées : {dict(enumerate(self.label_encoder.classes_))}")
        return df

    def split_data(self, df: pd.DataFrame) -> SplitData:
        """
        Split dataset into train/val/test using stratification on target.
        """
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset.")

        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=self.val_size, random_state=self.random_state, stratify=y_trainval
        )

        return SplitData(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test
        )

    def detect_feature_types(self, X: pd.DataFrame):
        """Detect numeric and categorical columns."""
        numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

        return numeric_features, categorical_features

    def _onehot_encoder_kwargs(self):
        """
        Retourne kwargs compatibles pour OneHotEncoder selon la version de sklearn.
        """
        # sklearn 1.2+ uses sparse_output, older versions use sparse
        try:
            # test signature by creating encoder with sparse_output
            _ = OneHotEncoder(sparse_output=False)
            return {"handle_unknown": "ignore", "sparse_output": False}
        except TypeError:
            return {"handle_unknown": "ignore", "sparse": False}

    def build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """Build ColumnTransformer with scaling for numeric and onehot for categorical."""
        numeric_features, categorical_features = self.detect_feature_types(X)

        numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

        ohe_kwargs = self._onehot_encoder_kwargs()
        categorical_transformer = Pipeline(
            steps=[("onehot", OneHotEncoder(**ohe_kwargs))]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ],
            remainder="drop",
            sparse_threshold=0
        )

        self.preprocessor = preprocessor
        return preprocessor

    def _build_feature_names_after_fit(self, X: pd.DataFrame):
        """
        Construit self.feature_names_ après fit du ColumnTransformer.
        Utilise get_feature_names_out si disponible, sinon reconstruit manuellement.
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor non construit/fitté.")

        # Tentative simple : utiliser get_feature_names_out si disponible
        try:
            names = self.preprocessor.get_feature_names_out(X.columns)
            self.feature_names_ = list(names)
            return
        except Exception:
            pass

        # Fallback manuel : concat numeric + onehot feature names
        names = []
        if self.numeric_features:
            names.extend(self.numeric_features)

        if self.categorical_features:
            # parcourir les transformateurs pour retrouver OneHotEncoder et ses catégories
            for name, transformer, cols in self.preprocessor.transformers_:
                if name == "cat":
                    # transformer est un Pipeline; récupérer le OneHotEncoder
                    try:
                        ohe = None
                        if hasattr(transformer, "named_steps"):
                            for step in transformer.named_steps.values():
                                if isinstance(step, OneHotEncoder):
                                    ohe = step
                                    break
                        elif isinstance(transformer, OneHotEncoder):
                            ohe = transformer

                        if ohe is None:
                            for c in cols:
                                names.append(c)
                        else:
                            cats = ohe.categories_
                            for col, cat_list in zip(cols, cats):
                                for cat in cat_list:
                                    names.append(f"{col}__{str(cat)}")
                    except Exception:
                        for c in cols:
                            names.append(c)

        self.feature_names_ = names

    def fit(self, X_train: pd.DataFrame) -> None:
        """Fit preprocessing pipeline on training data and compute feature names."""
        if self.preprocessor is None:
            self.build_preprocessor(X_train)
        self.preprocessor.fit(X_train)
        self._build_feature_names_after_fit(X_train)

    def transform(self, X: pd.DataFrame):
        """Transform dataset using fitted preprocessor and return a DataFrame with column names."""
        if self.preprocessor is None:
            raise ValueError("Preprocessor is not fitted yet. Call fit() first.")

        X_t = self.preprocessor.transform(X)

        # If transform returns sparse matrix, convert to array
        if hasattr(X_t, "toarray"):
            X_t = X_t.toarray()

        # Ensure feature names are available
        if self.feature_names_ is None:
            self._build_feature_names_after_fit(X)

        return pd.DataFrame(X_t, columns=self.feature_names_)

    def fit_transform_splits(self, split_data: SplitData) -> dict:
        """
        Fit on train then transform train/val/test.
        Returns dict with transformed DataFrames + y.
        """
        self.fit(split_data.X_train)

        X_train_t = self.transform(split_data.X_train)
        X_val_t   = self.transform(split_data.X_val)
        X_test_t  = self.transform(split_data.X_test)

        return {
            "X_train": X_train_t,
            "X_val":   X_val_t,
            "X_test":  X_test_t,
            "y_train": split_data.y_train.reset_index(drop=True),
            "y_val":   split_data.y_val.reset_index(drop=True),
            "y_test":  split_data.y_test.reset_index(drop=True),
        }

    def get_feature_names(self) -> list:
        if self.feature_names_ is None:
            raise ValueError("feature_names_ non calculé. Appelez fit() d'abord.")
        return list(self.feature_names_)
