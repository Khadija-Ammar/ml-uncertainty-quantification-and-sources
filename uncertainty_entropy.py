import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import xgboost as xgb

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    recall_score,
    precision_score,
    brier_score_loss,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.calibration import CalibratedClassifierCV

from preprocessor import Preprocessor
from model import GBUQClassifier, GBConfig


warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG
# =============================================================================

DATA_PATH = "bank-full.csv"   # mets le csv dans le même dossier que ce script
RESULTS_DIR = "results_entropy_gb"
RANDOM_STATE = 42

os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================================================================
# HELPERS
# =============================================================================

def save_plot(filename: str):
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Graph saved -> {path}")


def save_dataframe(df: pd.DataFrame, filename: str):
    path = os.path.join(RESULTS_DIR, filename)
    df.to_csv(path, index=False)
    print(f"CSV saved -> {path}")


def evaluate_probas(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "brier": brier_score_loss(y_true, y_proba),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "threshold": threshold,
    }


def binary_entropy(p: np.ndarray) -> np.ndarray:
    eps = 1e-10
    p = np.clip(p, eps, 1 - eps)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def make_dense_df(X, feature_names):
    if hasattr(X, "toarray"):
        X = X.toarray()
    return pd.DataFrame(X, columns=feature_names)


def clean_feature_names(names):
    cleaned = []
    for c in names:
        c = c.replace("num__", "")
        c = c.replace("cat__", "")
        cleaned.append(c)
    return cleaned


def fit_calibrator_prefit(fitted_estimator, X_cal, y_cal, method="sigmoid"):
    """
    Calibrate a classifier already fitted beforehand.
    Important: pass the INNER sklearn model, not the custom wrapper.
    """
    calibrator = CalibratedClassifierCV(fitted_estimator, method=method, cv="prefit")
    calibrator.fit(X_cal, y_cal)
    return calibrator


# =============================================================================
# 1. LOAD DATA
# =============================================================================

print("=== 1. LOAD DATA ===")
df = pd.read_csv(DATA_PATH, sep=";")
print("Raw dataset shape:", df.shape)


# =============================================================================
# 2. PREPROCESSING
# - same logic as your project pipeline
# - duration excluded for consistency / leakage avoidance
# =============================================================================

print("\n=== 2. PREPROCESSING ===")

preproc = Preprocessor(
    target_column="y",
    drop_columns=["duration"],
    test_size=0.2,
    val_size=0.2,
    random_state=RANDOM_STATE
)

df = preproc.drop_unwanted_columns(df)
df = preproc.encode_target(df)

splits = preproc.split_data(df)
transformed = preproc.fit_transform_splits(splits)

X_train = transformed["X_train"]
X_val   = transformed["X_val"]
X_test  = transformed["X_test"]

y_train = transformed["y_train"]
y_val   = transformed["y_val"]
y_test  = transformed["y_test"]

print("X_train shape:", X_train.shape)
print("X_val shape  :", X_val.shape)
print("X_test shape :", X_test.shape)

feature_names = preproc.preprocessor.get_feature_names_out()
feature_names = clean_feature_names(feature_names)

X_train_df = make_dense_df(X_train, feature_names)
X_val_df   = make_dense_df(X_val, feature_names)
X_test_df  = make_dense_df(X_test, feature_names)

print("Number of transformed features:", len(feature_names))


# =============================================================================
# 3. TRAIN GB CLASSIFIER
# =============================================================================

print("\n=== 3. TRAIN GB CLASSIFIER ===")

gb = GBUQClassifier(
    GBConfig(
        n_estimators=18,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=10,
        random_state=RANDOM_STATE,
        minority_weight=8.0
    )
)

gb.fit(X_train, y_train)

proba_train = gb.predict_proba(X_train)[:, 1]
proba_val   = gb.predict_proba(X_val)[:, 1]
proba_test  = gb.predict_proba(X_test)[:, 1]

gb_metrics = pd.DataFrame([
    {"split": "train", **evaluate_probas(y_train, proba_train, 0.5)},
    {"split": "val",   **evaluate_probas(y_val,   proba_val,   0.5)},
    {"split": "test",  **evaluate_probas(y_test,  proba_test,  0.5)},
])

print("\nGB classifier metrics:")
print(gb_metrics.to_string(index=False))
save_dataframe(gb_metrics, "gb_classifier_metrics.csv")


# =============================================================================
# 4. CALIBRATION
# IMPORTANT: calibrate the INNER sklearn model -> gb.model
# =============================================================================

print("\n=== 4. CALIBRATION ===")

cal_gb = fit_calibrator_prefit(gb.model, X_val, y_val, method="sigmoid")

proba_val_cal  = cal_gb.predict_proba(X_val)[:, 1]
proba_test_cal = cal_gb.predict_proba(X_test)[:, 1]

cal_metrics = pd.DataFrame([
    {"split": "val_calibrated",  **evaluate_probas(y_val,  proba_val_cal,  0.5)},
    {"split": "test_calibrated", **evaluate_probas(y_test, proba_test_cal, 0.5)},
])

print("\nCalibrated GB metrics:")
print(cal_metrics.to_string(index=False))
save_dataframe(cal_metrics, "gb_calibrated_metrics.csv")


# =============================================================================
# 5. ENTROPY UNCERTAINTY
# =============================================================================

print("\n=== 5. ENTROPY UNCERTAINTY ===")

entropy_val  = binary_entropy(proba_val_cal)
entropy_test = binary_entropy(proba_test_cal)

entropy_stats = pd.DataFrame([
    {
        "split": "val",
        "mean": float(entropy_val.mean()),
        "std":  float(entropy_val.std()),
        "min":  float(entropy_val.min()),
        "max":  float(entropy_val.max()),
    },
    {
        "split": "test",
        "mean": float(entropy_test.mean()),
        "std":  float(entropy_test.std()),
        "min":  float(entropy_test.min()),
        "max":  float(entropy_test.max()),
    }
])

print("\nEntropy statistics:")
print(entropy_stats.to_string(index=False))
save_dataframe(entropy_stats, "entropy_stats.csv")

plt.figure(figsize=(8, 5))
plt.hist(entropy_test, bins=50)
plt.xlabel("Entropy")
plt.ylabel("Count")
plt.title("Distribution of entropy uncertainty (test)")
plt.tight_layout()
save_plot("entropy_distribution_test.png")

plt.figure(figsize=(8, 5))
plt.scatter(proba_test_cal, entropy_test, alpha=0.35)
plt.xlabel("Calibrated predicted probability")
plt.ylabel("Entropy")
plt.title("Calibrated probability vs entropy")
plt.tight_layout()
save_plot("proba_vs_entropy_test.png")


# =============================================================================
# 6. U-MODEL : PREDICT ENTROPY FROM X
# Train on VAL, evaluate on TEST
# =============================================================================

print("\n=== 6. XGBOOST U-MODEL ===")

u_model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    random_state=RANDOM_STATE
)

u_model.fit(X_val_df, entropy_val)

entropy_pred_val  = u_model.predict(X_val_df)
entropy_pred_test = u_model.predict(X_test_df)

u_metrics = pd.DataFrame([
    {
        "split": "val",
        "RMSE": np.sqrt(mean_squared_error(entropy_val, entropy_pred_val)),
        "MAE":  mean_absolute_error(entropy_val, entropy_pred_val),
        "R2":   r2_score(entropy_val, entropy_pred_val),
    },
    {
        "split": "test",
        "RMSE": np.sqrt(mean_squared_error(entropy_test, entropy_pred_test)),
        "MAE":  mean_absolute_error(entropy_test, entropy_pred_test),
        "R2":   r2_score(entropy_test, entropy_pred_test),
    }
])

print("\nU-model metrics:")
print(u_metrics.to_string(index=False))
save_dataframe(u_metrics, "u_model_metrics.csv")

plt.figure(figsize=(8, 5))
plt.scatter(entropy_test, entropy_pred_test, alpha=0.35)
plt.xlabel("True entropy")
plt.ylabel("Predicted entropy")
plt.title("U-model: true vs predicted entropy")
plt.tight_layout()
save_plot("true_vs_predicted_entropy.png")


# =============================================================================
# 7. SHAP ANALYSIS
# =============================================================================

print("\n=== 7. SHAP ANALYSIS ===")

explainer = shap.Explainer(u_model, X_test_df)
shap_values = explainer(X_test_df)

# Bar plot
plt.figure()
shap.plots.bar(shap_values, show=False)
save_plot("shap_bar_entropy_gb.png")

# Beeswarm plot
plt.figure()
shap.plots.beeswarm(shap_values, show=False)
save_plot("shap_beeswarm_entropy_gb.png")

# Ranking table
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
shap_importance_df = pd.DataFrame({
    "feature": X_test_df.columns,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False)

print("\nTop 15 features driving entropy uncertainty:")
print(shap_importance_df.head(15).to_string(index=False))
save_dataframe(shap_importance_df, "shap_importance_entropy_gb.csv")


# =============================================================================
# 8. HIGH UNCERTAINTY ANALYSIS
# =============================================================================

print("\n=== 8. HIGH UNCERTAINTY ANALYSIS ===")

threshold = np.quantile(entropy_test, 0.90)
high_uncertainty = entropy_test >= threshold

analysis_df = X_test_df.copy()
analysis_df["y_true"] = np.array(y_test)
analysis_df["proba_calibrated"] = proba_test_cal
analysis_df["entropy"] = entropy_test
analysis_df["pred_entropy"] = entropy_pred_test
analysis_df["high_uncertainty"] = high_uncertainty.astype(int)
analysis_df["gb_pred_class"] = (proba_test_cal >= 0.5).astype(int)
analysis_df["gb_error"] = (analysis_df["gb_pred_class"] != analysis_df["y_true"]).astype(int)

print(f"High uncertainty threshold (90th percentile): {threshold:.6f}")
print(f"Share of high-uncertainty cases: {analysis_df['high_uncertainty'].mean():.4f}")

error_rate_by_group = analysis_df.groupby("high_uncertainty", as_index=False)["gb_error"].mean()
error_rate_by_group["group"] = error_rate_by_group["high_uncertainty"].map({
    0: "Low uncertainty",
    1: "High uncertainty"
})

print("\nError rate by uncertainty group:")
print(error_rate_by_group[["group", "gb_error"]].to_string(index=False))

save_dataframe(error_rate_by_group, "error_rate_by_uncertainty_group.csv")
save_dataframe(analysis_df, "entropy_analysis_test_table.csv")

plt.figure(figsize=(7, 5))
plt.bar(error_rate_by_group["group"], error_rate_by_group["gb_error"])
plt.ylabel("Classification error rate")
plt.title("GB error rate by uncertainty group")
plt.tight_layout()
save_plot("error_rate_by_uncertainty_group.png")


# =============================================================================
# 9. TOP FEATURE COMPARISON
# =============================================================================

print("\n=== 9. TOP FEATURE COMPARISON ===")

top_features = shap_importance_df["feature"].head(5).tolist()
comparison_rows = []

for feat in top_features:
    low_mean = analysis_df.loc[analysis_df["high_uncertainty"] == 0, feat].mean()
    high_mean = analysis_df.loc[analysis_df["high_uncertainty"] == 1, feat].mean()
    comparison_rows.append({
        "feature": feat,
        "mean_low_uncertainty": low_mean,
        "mean_high_uncertainty": high_mean,
        "difference_high_minus_low": high_mean - low_mean
    })

comparison_df = pd.DataFrame(comparison_rows)
print(comparison_df.to_string(index=False))
save_dataframe(comparison_df, "top_features_high_vs_low_uncertainty.csv")


# =============================================================================
# 10. SHAP DEPENDENCE PLOTS
# =============================================================================

print("\n=== 10. SHAP DEPENDENCE PLOTS ===")

for feat in top_features[:3]:
    plt.figure()
    shap.dependence_plot(
        feat,
        shap_values.values,
        X_test_df,
        show=False
    )
    safe_feat = feat.replace("/", "_").replace(" ", "_")
    save_plot(f"shap_dependence_{safe_feat}.png")


# =============================================================================
# END
# =============================================================================

print("\n=== DONE ===")
print(f"All outputs saved in: {RESULTS_DIR}")