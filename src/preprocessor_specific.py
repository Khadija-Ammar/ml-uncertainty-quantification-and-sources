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

    SKEWED_VARS       = ["balance", "campaign", "previous"]
    CATEGORICAL_NOMINAL = ["job", "marital", "education", "contact", "month", "poutcome"]
    CATEGORICAL_BINARY  = ["default", "housing", "loan"]
    NUMERICAL_FEATURES  = ["age", "balance", "day", "campaign", "previous",
                           "pdays_contacted", "pdays_positive"]

    def __init__(
        self,
        target_column: str = "deposit",
        drop_columns: list = None,
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 42,
    ):
        self.target_column = target_column
        self.drop_columns  = drop_columns if drop_columns is not None else []
        self.test_size     = test_size
        self.val_size      = val_size
        self.random_state  = random_state
        self.preprocessor  = None
        self.label_encoder = LabelEncoder()

    # ------------------------------------------------------------------
    # 0. I/O
    # ------------------------------------------------------------------

    def load_data(self, path: str, sep: str = ";") -> pd.DataFrame:
        return pd.read_csv(path, sep=sep)

    # ------------------------------------------------------------------
    # 1. Column cleaning
    # ------------------------------------------------------------------

    def drop_unwanted_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=self.drop_columns, errors="ignore")

    # ------------------------------------------------------------------
    # 2. Target encoding
    # ------------------------------------------------------------------

    def encode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found.")
        df = df.copy()
        df[self.target_column] = self.label_encoder.fit_transform(df[self.target_column])
        print(f"[encode_target] Classes: {dict(enumerate(self.label_encoder.classes_))}")
        return df

    # ------------------------------------------------------------------
    # 3. Train / val / test split (stratified)
    # ------------------------------------------------------------------

    def split_data(self, df: pd.DataFrame) -> SplitData:
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=self.test_size,          # 0.20
            random_state=self.random_state, stratify=y,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=0.25,  # ← 0.25 × 0.80 = 0.20 total
            random_state=self.random_state, stratify=y_trainval,
        )
        return SplitData(X_train=X_train, X_val=X_val, X_test=X_test,
                        y_train=y_train, y_val=y_val, y_test=y_test)

    # ------------------------------------------------------------------
    # 4. Cas spécial : pdays
    # ------------------------------------------------------------------

    @staticmethod
    def process_pdays(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["pdays_contacted"] = (df["pdays"] != -1).astype(int)
        df["pdays_positive"]  = df["pdays"].clip(lower=0)
        df = df.drop(columns=["pdays"])
        return df

    # ------------------------------------------------------------------
    # 5. Skewed numeric features — signed log1p
    # ------------------------------------------------------------------

    @staticmethod
    def signed_log1p(x: pd.Series) -> pd.Series:
        return np.sign(x) * np.log1p(np.abs(x))

    def apply_signed_log(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.SKEWED_VARS:
            if col in df.columns:
                df[col] = self.signed_log1p(df[col])
        return df

    # ------------------------------------------------------------------
    # 6. ColumnTransformer
    # ------------------------------------------------------------------

        
    def build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        numerical   = [c for c in self.NUMERICAL_FEATURES    if c in X.columns]
        cat_nominal = [c for c in self.CATEGORICAL_NOMINAL   if c in X.columns]
        cat_binary  = [c for c in self.CATEGORICAL_BINARY    if c in X.columns]

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num",     StandardScaler(),                                                                     numerical),
                ("cat_nom", OneHotEncoder(drop="first",     handle_unknown="ignore", sparse_output=False),       cat_nominal),
                ("cat_bin", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False),       cat_binary),
            ],
            remainder="drop",
        ).set_output(transform="pandas")

        return self.preprocessor

    # ------------------------------------------------------------------
    # 7. Fit / transform
    # ------------------------------------------------------------------

    def _custom_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self.process_pdays(X)
        X = self.apply_signed_log(X)
        return X

    def fit(self, X_train: pd.DataFrame) -> None:
        X_custom = self._custom_transform(X_train)
        if self.preprocessor is None:
            self.build_preprocessor(X_custom)
        self.preprocessor.fit(X_custom)

    def transform(self, X: pd.DataFrame):
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted yet. Call fit() first.")
        return self.preprocessor.transform(self._custom_transform(X))

    def fit_transform_splits(self, split_data: SplitData) -> dict:
        self.fit(split_data.X_train)

        X_train_t = self.transform(split_data.X_train)
        X_val_t   = self.transform(split_data.X_val)
        X_test_t  = self.transform(split_data.X_test)

        print(f"Proportion y_train : {split_data.y_train.mean():.4f}")
        print(f"Proportion y_val   : {split_data.y_val.mean():.4f}")
        print(f"Proportion y_test  : {split_data.y_test.mean():.4f}")
        print(f"Shapes: {X_train_t.shape}, {X_val_t.shape}, {X_test_t.shape}")

        return {
            "X_train": X_train_t, "X_val": X_val_t, "X_test": X_test_t,
            "y_train": split_data.y_train,
            "y_val":   split_data.y_val,
            "y_test":  split_data.y_test,
        }

    def get_feature_names_out(self) -> list:
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted yet. Call fit() first.")
        return list(self.preprocessor.get_feature_names_out())