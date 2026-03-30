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
from conformal import (quantile, fit_split_cp_proba, predict_sets_from_tau, coverage_efficiency, compute_U, distribution,test1,test2, eda_u_distribution, eda_u_vs_proba, eda_u_by_group)
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

df = preproc.load_data("data/bank-full.csv", sep=";")   # ✅ Chargement
df = preproc.drop_unwanted_columns(df)             # ✅ Nettoyage
df = preproc.encode_target(df)                   # ✅ Encodage cible
splits = preproc.split_data(df)                    # ✅ Split stratifié
transformed_data = preproc.fit_transform_splits(splits)  # ✅ Fit + Transform
df_test_raw = splits.X_test.copy()
df_test_raw["y"] = splits.y_test.values

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





# ========================= Khadija
# Split Conformal Prediction sur Logistic regression
# =========================

alpha = 0.10  # => couverture cible 90%

cp_model = lr   

# 1) Calibration conformal sur VAL
cp_fit = fit_split_cp_proba(cp_model, X_val, y_val, alpha=alpha)
tau = cp_fit["tau"]
print("\n=== Split CP calibration ===")
print(f"alpha={alpha} | qhat={cp_fit['qhat']:.4f} | tau={tau:.4f}")

# 2) Sets + U sur TEST
uq = compute_U(cp_model, X_test, tau=tau)
pred_sets = uq["pred_sets"]
U = uq["U"]

# 3) Coverage & efficiency sur TEST
cp_metrics = coverage_efficiency(pred_sets, y_test)
print("\n=== Split CP evaluation (TEST) ===")
print(f"Target coverage ~ {1-alpha:.2f}")
print(f"Coverage      = {cp_metrics['coverage']:.4f}")
print(f"Avg set size  = {cp_metrics['avg_set_size']:.4f}") 

n = len(U)
p_u1 = (U == 1).mean()
p_u2 = (U == 2).mean()
# 4) Distribution de U et qq stat desc

print("\n=== Uncertainty U = |Gamma(x)| ===")
print(f"P(U=1) = {p_u1:.4f}  ({p_u1*n:.0f}/{n})")
print(f"P(U=2) = {p_u2:.4f}  ({p_u2*n:.0f}/{n})")
print(f"Avg(U) = {U.mean():.4f}")

u_summary = eda_u_distribution(U)

u_proba_summary = eda_u_vs_proba(cp_model, X_test, U)

u_group_summary = eda_u_by_group(
    df_test_raw=df_test_raw,
    U=U,
    group_cols=["job", "contact", "poutcome", "marital", "education"]
)

# 5) tests d'hypothèses : U(x)=2 plus fréquent quand p1(x) est proche de 0.5 / Différences par sous-groupes (Où l’incertitude est-elle systématiquement plus élevée ?)

test1(cp_model, X_test, U)

p1_test = cp_model.predict_proba(X_test)[:, 1]

results_tests = test2(
    df_test_raw=df_test_raw,
    U=U,
    p1=p1_test,
    y_true=y_test,
    group_cols=["job", "contact", "poutcome", "marital", "education"]
)
# ====================== Khadija 
# Cadre Logistic Regression et U discrète : 
# Modélisation de U avec XGBoost puis calcul des valeurs de shapley
# ======================
from shap_lr_discrete import (get_uncertainty_labels, train_and_evaluate_xgb, 
                        prepare_shap_data, compute_and_plot_all_shap, 
                        plot_top_dependencies, get_full_importance_table, get_most_uncertain_obs)

# ... (après avoir calculé U_train et U_test avec conformal.py) ...

# 1. Labels
y_u_train, y_u_test = get_uncertainty_labels(cp_model, X_train, X_test, tau)

# 2. DataFrames
feature_names = preproc.get_feature_names_out()
X_train_df, X_test_df = prepare_shap_data(X_train, X_test, feature_names)

# 3. Training + Éval complète 
xgb_u = train_and_evaluate_xgb(X_train_df, y_u_train, X_test_df, y_u_test)

# 4. SHAP Plots 
shap_values = compute_and_plot_all_shap(xgb_u, X_test_df)

# 5. Dependence Plots 
plot_top_dependencies(shap_values, X_test_df)

# 6. Tableau d'importance
importance_table = get_full_importance_table(shap_values, X_test_df)
print("\n=== Top 20 features selon |SHAP| moyen ===")
print(importance_table.head(20))

# 7. Obs les plus incertaines 
top_uncertain = get_most_uncertain_obs(X_test_df, y_u_test, xgb_u)
print("\n=== Top 10 des observations les plus incertaines ===")
print(top_uncertain)
