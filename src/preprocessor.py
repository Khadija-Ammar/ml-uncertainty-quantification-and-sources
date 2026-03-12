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
    Preprocessor class that replicates the preprocessing steps from the notebook.

    Main responsibilities:
    - load data
    - clean / drop columns
    - encode target column
    - split train/val/test (stratified)
    - build preprocessing pipeline (scaling + encoding)
    - fit and transform splits
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
        # ✅ Ajout : LabelEncoder pour la variable cible
        self.label_encoder = LabelEncoder()


    def drop_unwanted_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns that should not be used."""
        return df.drop(columns=self.drop_columns, errors="ignore")

    def encode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode la variable cible en valeurs numériques (0/1).
        Ex : 'yes' -> 1, 'no' -> 0.
        Le LabelEncoder est fitté ici et pourra être réutilisé.
        """
        if self.target_column not in df.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found in dataset."
            )

        df = df.copy()

        # ✅ Fit + transform sur toute la colonne cible AVANT le split
        df[self.target_column] = self.label_encoder.fit_transform(
            df[self.target_column]
        )

        print(
            f"[encode_target] Classes encodées : "
            f"{dict(enumerate(self.label_encoder.classes_))}"
        )

        return df

    def split_data(self, df: pd.DataFrame) -> SplitData:
        """
        Split dataset into train/val/test using stratification on target.
        val_size is applied on train split (after test split).
        """
        if self.target_column not in df.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found in dataset."
            )

        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval,
            y_trainval,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=y_trainval
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
        numeric_features = X.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()
        categorical_features = X.select_dtypes(
            include=["object"]
        ).columns.tolist()

        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

        return numeric_features, categorical_features

    def build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """Build ColumnTransformer with scaling for numeric and onehot for categorical."""
        numeric_features, categorical_features = self.detect_feature_types(X)

        numeric_transformer = Pipeline(
            steps=[("scaler", StandardScaler())]
        )

        categorical_transformer = Pipeline(
            steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        self.preprocessor = preprocessor
        return preprocessor

    def fit(self, X_train: pd.DataFrame) -> None:
        """Fit preprocessing pipeline on training data."""
        if self.preprocessor is None:
            self.build_preprocessor(X_train)
        self.preprocessor.fit(X_train)

    def transform(self, X: pd.DataFrame):
        """Transform dataset using fitted preprocessor."""
        if self.preprocessor is None:
            raise ValueError("Preprocessor is not fitted yet. Call fit() first.")
        return self.preprocessor.transform(X)

    def fit_transform_splits(self, split_data: SplitData) -> dict:
        """
        Fit on train then transform train/val/test.
        Returns dict with transformed matrices + y.
        """
        self.fit(split_data.X_train)

        X_train_t = self.transform(split_data.X_train)
        X_val_t   = self.transform(split_data.X_val)
        X_test_t  = self.transform(split_data.X_test)

        return {
            "X_train": X_train_t,
            "X_val":   X_val_t,
            "X_test":  X_test_t,
            "y_train": split_data.y_train,
            "y_val":   split_data.y_val,
            "y_test":  split_data.y_test,
        }


    

    
