from preprocessor import Preprocessor
from model import (
    GBUQClassifier,  LRUQClassifier,
    GBConfig,        LRConfig,
    plot_roc_curves  
)
from calibrator import prob_calibrator, expected_calibration_error, plot_ece_comparison, brier_score_loss, plot_brier_score_comparison
from inference_prop_pred import (
    plot_proba_distributions,
    ks_test_analysis,
)
from CP_Splitor import SplitConformalClassifier
import numpy as np
import pandas as pd


#=============================================================================
# 1. PREPROCESSING 
#=============================================================================

print ( "=== 1. PREPROCESSING ===" )

preproc = Preprocessor(                 # ✅ Initialisation avec les bons paramètres
    target_column="y",
    drop_columns=["duration"],
    test_size=0.2,
    val_size=0.2,
    random_state=42
)

df = pd.read_csv("data\\bank-full.csv", sep=";")   # ✅ Chargement
print ( "dimension du dataset :", df.shape)
df = preproc.drop_unwanted_columns(df)             # ✅ Nettoyage
print ( "dimension après nettoyage :", df.shape)
df = preproc.encode_target(df)                   # ✅ Encodage cible
print ( "dimension après encodage :", df.shape)
splits = preproc.split_data(df)     # ✅ Split stratifié
transformed_data = preproc.fit_transform_splits(splits)  # ✅ Fit + Transform

X_train = transformed_data["X_train"]
X_val   = transformed_data["X_val"]
X_test  = transformed_data["X_test"]

print ( "dimension X_train :", X_train.shape)
print ( "dimension X_val :", X_val.shape)   
print ( "dimension X_test :", X_test.shape)

y_train = transformed_data["y_train"]
y_val   = transformed_data["y_val"]
y_test  = transformed_data["y_test"]    

print ( "dimension y_train :", y_train.shape)
print ( "dimension y_val :", y_val.shape)
print ( "dimension y_test :", y_test.shape)

print("Preprocessing complete.")


#=============================================================================
# 2. MODEL TRAINING
#=============================================================================
print ( "\n=== 2. MODEL TRAINING ===" )

gb = GBUQClassifier(GBConfig(minority_weight=8.0))
gb.fit(X_train, y_train)

lr = LRUQClassifier(LRConfig(class_weight={0: 1.0, 1: 8.0}))
lr.fit(X_train, y_train)

#  Évaluation sur validation (avant calibration)

print("\n--- Évaluation sur VALIDATION (non calibré) ---")
gb.evaluate(X_val, y_val, label="GB  non calibré")
lr.evaluate(X_val, y_val, label="LR  non calibré")


# Courbes ROC sur val set ( non calibré)
plot_roc_curves(
    classifiers={
        "GB  non calibré" : gb,   # use_calibrated=True par défaut donc...
        "LR  non calibré" : lr,
    },
    X=X_val, y=y_val
)

# --- Calibration (sur validation set) ---
cal_gb = prob_calibrator(gb, X_val, y_val, method="sigmoid")
cal_lr = prob_calibrator(lr, X_val, y_val, method="sigmoid")

# --- ECE sur test set ---
ece_dict = {
    "GB non calibré" : expected_calibration_error(y_val, gb.predict_proba(X_val)[:, 1]),
    "GB calibré"     : expected_calibration_error(y_val, cal_gb.predict_proba(X_val)[:, 1]),
    "LR non calibré" : expected_calibration_error(y_val, lr.predict_proba(X_val)[:, 1]),
    "LR calibré"     : expected_calibration_error(y_val, cal_lr.predict_proba(X_val)[:, 1]),
}
plot_ece_comparison(ece_dict)

#  brier score sur le val set
brier_dict = {
    "GB non calibré" : brier_score_loss(y_val, gb.predict_proba(X_val)[:, 1]),
    "GB calibré"     : brier_score_loss(y_val, cal_gb.predict_proba(X_val)[:, 1]),
    "LR non calibré" : brier_score_loss(y_val, lr.predict_proba(X_val)[:, 1]),
    "LR calibré"     : brier_score_loss(y_val, cal_lr.predict_proba(X_val)[:, 1]),
}
plot_brier_score_comparison(brier_dict)

print("Model training complete.")




# ==============================================================================
# 3. Split conformal prediction on test set
# =============================================================================
print ( "\n=== 3. Split conformal prediction on TEST set ===" )

# CP sur GB (calibré )
cp_gb = SplitConformalClassifier(alpha=0.1)
cp_gb.calibrate(cal_gb, X_val, y_val)          # ou cal_gb si calibré

pred_sets_gb = cp_gb.predict_set(cal_gb, X_test)
#metrics_gb   = cp_gb.compute_metrics(pred_sets_gb, y_test)

# Visualisations
cp_gb.plot_nonconformity_scores()
cp_gb.plot_prediction_sets(pred_sets_gb, y_test)
#cp_gb.plot_coverage_vs_alpha(cal_gb, X_val, y_val, X_test, y_test)

# CP sur LR (calibré)
cp_lr = SplitConformalClassifier(alpha=0.1)
cp_lr.calibrate(cal_lr, X_val, y_val)

pred_sets_lr = cp_lr.predict_set(cal_lr, X_test)
#metrics_lr   = cp_lr.compute_metrics(pred_sets_lr, y_test)

cp_lr.plot_nonconformity_scores()
cp_lr.plot_prediction_sets(pred_sets_lr, y_test)
#cp_lr.plot_coverage_vs_alpha(cal_lr, X_val, y_val, X_test, y_test)