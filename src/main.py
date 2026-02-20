from preprocessor import Preprocessor
from model import (
    GBUQClassifier,  LRUQClassifier,
    GBConfig,        LRConfig,
    plot_roc_curves  
)
from calibrator import prob_calibrator, expected_calibration_error, plot_ece_comparison
from inference_prop_pred import (
    plot_proba_distributions,
    ks_test_analysis,
)
import numpy as np


#***************************
# 1. PREPROCESSING 
#***************************


preproc = Preprocessor(                 # ✅ Initialisation avec les bons paramètres
    target_column="y",
    drop_columns=["duration"],
    test_size=0.2,
    val_size=0.2,
    random_state=42
)

df = preproc.load_data("data/bank.csv", sep=";")   # ✅ Chargement
df = preproc.drop_unwanted_columns(df)             # ✅ Nettoyage
df = preproc.encode_target(df)                   # ✅ Encodage cible
splits = preproc.split_data(df)                    # ✅ Split stratifié
transformed_data = preproc.fit_transform_splits(splits)  # ✅ Fit + Transform


X_train = transformed_data["X_train"]
X_val   = transformed_data["X_val"]
X_test  = transformed_data["X_test"]

y_train = transformed_data["y_train"]
y_val   = transformed_data["y_val"]
y_test  = transformed_data["y_test"]



print("Preprocessing complete.")


#*******************
# 2. MODEL TRAINING
#*******************


gb = GBUQClassifier(GBConfig(minority_weight=8.0))
gb.fit(X_train, y_train)

lr = LRUQClassifier(LRConfig(class_weight={0: 1.0, 1: 8.0}))
lr.fit(X_train, y_train)

# ==============================================================================
# 3. Évaluation sur validation (avant calibration)
# ==============================================================================
print("\n--- Évaluation sur VALIDATION (non calibré) ---")
gb.evaluate(X_val, y_val, label="GB  non calibré")
lr.evaluate(X_val, y_val, label="LR  non calibré")

# ==============================================================================
# 4. Courbes ROC sur val set ( non calibré)
# ==============================================================================
plot_roc_curves(
    classifiers={
        "GB  non calibré" : gb,   # use_calibrated=True par défaut donc...
        "LR  non calibré" : lr,
    },
    X=X_val, y=y_val
)

# ==============================================================================
# 5. Calibration sur validation set
# ==============================================================================

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

"""# ==============================================================================
# 6. Evaluation final du model sur test set (calibré vs non calibré)
# ==============================================================================
print("\n--- Évaluation sur TEST SET ---")
for label, clf in [
    ("GB non calibré", gb),
    
    ("LR non calibré", lr),
    
]:
    clf.evaluate(X_test, y_test, label=label)
    
plot_roc_curves(
    classifiers={
        
        "
        "GB non calibré" : gb,
        "LR non calibré" : lr,
    },
    X=X_test, y=y_test
)   
"""
# ==============================================================================
# 7. Analyse des distributions de probabilités prédites
# ==============================================================================

# Les classifiers peuvent être custom ou calibrés
all_models = {
    "GB non calibré" : gb,       # GBUQClassifier
    "LR non calibré" : lr,       # LRUQClassifier
    "GB calibré"     : cal_gb,   # CalibratedClassifierCV
    "LR calibré"     : cal_lr,   # CalibratedClassifierCV
}

# Distribution des probabilités
plot_proba_distributions(all_models, X_test, title_suffix="(test set)")

# Test KS + visualisation
ks_results = ks_test_analysis(all_models, X_test)

