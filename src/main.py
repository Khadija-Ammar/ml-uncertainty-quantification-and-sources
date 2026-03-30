from preprocessor import Preprocessor
from model import (
    GBUQClassifier,  LRUQClassifier,
    GBConfig,        LRConfig,
    plot_roc_curves
)
from calibrator import prob_calibrator, expected_calibration_error, plot_ece_comparison, brier_score_loss, plot_brier_score_comparison, plot_BA_calibration_curve
from inference_prop_pred import (
    plot_proba_distributions,
    ks_test_analysis,
)
from CP_Splitor import SplitConformalClassifier
from Interpretor import UncertaintyInterpreter


import numpy as np
import pandas as pd
import os

import matplotlib
matplotlib.use("Agg")   # empêche l'ouverture des fenêtres de graph
import matplotlib.pyplot as plt


# =============================================================================
# RESULTS DIRECTORY
# =============================================================================

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_plot(name):
    """Sauvegarde un graphique matplotlib dans results/"""
    path = os.path.join(RESULTS_DIR, name)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Graph saved -> {path}")


# =============================================================================
# 1. PREPROCESSING
# =============================================================================

print("=== 1. PREPROCESSING ===")

preproc = Preprocessor(
    target_column="y",
    drop_columns=["duration"],
    test_size=0.2,
    val_size=0.2,
    random_state=42
)

df = pd.read_csv("data\\bank-full.csv", sep=";")

print("dimension du dataset :", df.shape)

df = preproc.drop_unwanted_columns(df)
print("dimension après nettoyage :", df.shape)

df = preproc.encode_target(df)
print("dimension après encodage :", df.shape)

splits = preproc.split_data(df)
transformed_data = preproc.fit_transform_splits(splits)

X_train = transformed_data["X_train"]
X_val   = transformed_data["X_val"]
X_test  = transformed_data["X_test"]

print("dimension X_train :", X_train.shape)
print("dimension X_val :", X_val.shape)
print("dimension X_test :", X_test.shape)

y_train = transformed_data["y_train"]
y_val   = transformed_data["y_val"]
y_test  = transformed_data["y_test"]

print("dimension y_train :", y_train.shape)
print("dimension y_val :", y_val.shape)
print("dimension y_test :", y_test.shape)

print("Preprocessing complete.")


# =============================================================================
# 2. MODEL TRAINING
# =============================================================================

print("\n=== 2. MODEL TRAINING ===")

gb = GBUQClassifier(GBConfig(minority_weight=8.0))
gb.fit(X_train, y_train)

lr = LRUQClassifier(LRConfig(class_weight={0: 1.0, 1: 8.0}))
lr.fit(X_train, y_train)


# Évaluation sur validation

print("\n--- Évaluation sur VALIDATION (non calibré) ---")

gb.evaluate(X_val, y_val, label="GB non calibré")
lr.evaluate(X_val, y_val, label="LR non calibré")


# ROC CURVES

plot_roc_curves(
    classifiers={
        "GB non calibré": gb,
        "LR non calibré": lr,
    },
    X=X_val,
    y=y_val
)

save_plot("roc_curves_validation.png")


# =============================================================================
# CALIBRATION
# =============================================================================

cal_gb = prob_calibrator(gb, X_val, y_val, method="sigmoid")
cal_lr = prob_calibrator(lr, X_val, y_val, method="sigmoid")

# Before/After calibration curve
plot_BA_calibration_curve(gb, cal_gb, X_val, y_val)
save_plot("calibration_curve_gb.png")

plot_BA_calibration_curve(lr, cal_lr, X_val, y_val)
save_plot("calibration_curve_lr.png")

# ECE

ece_dict = {
    "GB non calibré": expected_calibration_error(y_val, gb.predict_proba(X_val)[:, 1]),
    "GB calibré": expected_calibration_error(y_val, cal_gb.predict_proba(X_val)[:, 1]),
    "LR non calibré": expected_calibration_error(y_val, lr.predict_proba(X_val)[:, 1]),
    "LR calibré": expected_calibration_error(y_val, cal_lr.predict_proba(X_val)[:, 1]),
}

plot_ece_comparison(ece_dict)
save_plot("ece_comparison.png")


# BRIER SCORE

brier_dict = {
    "GB non calibré": brier_score_loss(y_val, gb.predict_proba(X_val)[:, 1]),
    "GB calibré": brier_score_loss(y_val, cal_gb.predict_proba(X_val)[:, 1]),
    "LR non calibré": brier_score_loss(y_val, lr.predict_proba(X_val)[:, 1]),
    "LR calibré": brier_score_loss(y_val, cal_lr.predict_proba(X_val)[:, 1]),
}

plot_brier_score_comparison(brier_dict)
save_plot("brier_scores.png")

print("Model training complete.")


# =============================================================================
# 3. SPLIT CONFORMAL PREDICTION
# =============================================================================

print("\n=== 3. Split conformal prediction on TEST set ===")


# --- GB MODEL ---

cp_gb = SplitConformalClassifier(alpha=0.1)
cp_gb.calibrate(cal_gb, X_val, y_val)

pred_sets_gb = cp_gb.predict_set(model=cal_gb, X_test=X_test)

cp_gb.plot_nonconformity_scores()
save_plot("gb_nonconformity_scores.png")

cp_gb.plot_prediction_sets(pred_sets_gb, y_test)
save_plot("gb_prediction_sets.png")


# --- LR MODEL ---

cp_lr = SplitConformalClassifier(alpha=0.1)
cp_lr.calibrate(cal_lr, X_val, y_val)

pred_sets_lr = cp_lr.predict_set(model=cal_lr, X_test=X_test)

cp_lr.plot_nonconformity_scores()
save_plot("lr_nonconformity_scores.png")

cp_lr.plot_prediction_sets(pred_sets_lr, y_test)
save_plot("lr_prediction_sets.png")


# =============================================================================
# UNCERTAINTY ANALYSIS
# =============================================================================

print("\n=== 4. UNCERTAINTY ANALYSIS ===")

# Ajout des incertitudes sur X_val
df_for_uncert_gb = cp_gb.add_uncertainties_to_dataset(model=cal_gb, X=X_test)
df_for_uncert_lr = cp_lr.add_uncertainties_to_dataset(model=cal_lr, X=X_test)

# Tracé des distributions d'incertitude
cp_gb.plot_float_uncertainty_distribution()
save_plot("uncertainty_entropy_gb.png")

cp_gb.plot_height_uncertainty_distribution()
save_plot("uncertainty_height_gb.png")

cp_lr.plot_float_uncertainty_distribution()
save_plot("uncertainty_entropy_lr.png")

cp_lr.plot_height_uncertainty_distribution()
save_plot("uncertainty_height_lr.png")


print("\n=== PIPELINE COMPLETE ===")
print(f"All graphs saved in : {RESULTS_DIR}/")

#====================================================================================
# Uncertainty Interpretability
#====================================================================================


interp = UncertaintyInterpreter()

interp.fit_entropy_model(df_for_uncert_gb)
interp.fit_height_model(df_for_uncert_gb)

interp.compute_shap_entropy(df_for_uncert_gb)
interp.compute_shap_height(df_for_uncert_gb)

interp.plot_shap_entropy(df_for_uncert_gb)
save_plot("shap_entropy.png")
interp.plot_shap_height(df_for_uncert_gb)
save_plot("shap_height.png")

top_entropy, top_height = interp.get_top_features(df_for_uncert_gb, k=10)

print("\nTop features générant de l'entropie :")
print(top_entropy)

print("\nTop features générant de la height uncertainty :")
print(top_height)
