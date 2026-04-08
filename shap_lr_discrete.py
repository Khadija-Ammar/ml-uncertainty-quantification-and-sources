import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

def get_uncertainty_labels(clf, X_train, X_test, tau):
    from conformal import compute_U
    res_train = compute_U(clf, X_train, tau=tau)
    res_test = compute_U(clf, X_test, tau=tau)
    y_u_train = (res_train["U"] == 2).astype(int)
    y_u_test = (res_test["U"] == 2).astype(int)
    return y_u_train, y_u_test

def prepare_shap_data(X_train, X_test, feature_names):
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    return X_train_df, X_test_df

def train_and_evaluate_xgb(X_train_df, y_u_train, X_test_df, y_u_test):
    # Calcul du scale_pos_weight
    spw = np.sum(y_u_train == 0) / np.sum(y_u_train == 1)
    
    xgb_u = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        eval_metric="logloss", scale_pos_weight=spw
    )
    xgb_u.fit(X_train_df, y_u_train)
    
    # --- Évaluation ---
    y_pred = xgb_u.predict(X_test_df)
    y_proba = xgb_u.predict_proba(X_test_df)[:, 1]
    
    print("\n=== Performance du XGBoost pour prédire U ===")
    print(f"Accuracy         : {accuracy_score(y_u_test, y_pred):.4f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_u_test, y_pred):.4f}")
    print(f"ROC-AUC          : {roc_auc_score(y_u_test, y_proba):.4f}")
    print("\n=== Classification report ===")
    print(classification_report(y_u_test, y_pred, digits=4))
    
    # Matrice de confusion
    cm = confusion_matrix(y_u_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    plt.figure(figsize=(6, 5))
    disp.plot()
    plt.title("Matrice de confusion — prédiction de U")
    plt.show(block=True)
    
    return xgb_u

def compute_and_plot_all_shap(model, X_df, max_display=20):
    explainer = shap.Explainer(model, X_df)
    shap_values = explainer(X_df)
    
    # 1. Beeswarm Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_df, max_display=max_display, show=False)
    plt.title("SHAP summary plot (beeswarm)")
    plt.show()
    
    # 2. Bar Plot 
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_df, plot_type="bar", max_display=max_display, show=False)
    plt.title("SHAP importance ranking — Top features expliquant U")
    plt.show()
    
    return shap_values

def plot_top_dependencies(shap_values, X_df, top_n=5):
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    indices = np.argsort(mean_abs_shap)[::-1][:top_n]
    top_features = X_df.columns[indices].tolist()
    
    print(f"\nTop features pour dependence plots : {top_features}")
    
    for feat in top_features:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        shap.dependence_plot(feat, shap_values.values, X_df, 
                             interaction_index="auto", ax=ax, show=False)
        
        plt.title(f"SHAP Dependence Plot: {feat}")
        plt.tight_layout()
        

        plt.show(block=True)

def get_full_importance_table(shap_values, X_df):
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    df_imp = pd.DataFrame({
        "feature": X_df.columns,
        "mean_abs_shap": mean_abs_shap
    }).sort_values(by="mean_abs_shap", ascending=False)
    df_imp["rank"] = range(1, len(df_imp) + 1)
    return df_imp

def get_most_uncertain_obs(X_df, y_u_test, model, n=10):
    results_u = X_df.copy()
    results_u["U_true"] = y_u_test
    results_u["U_pred"] = model.predict(X_df)
    results_u["proba_uncertainty"] = model.predict_proba(X_df)[:, 1]
    return results_u.sort_values(by="proba_uncertainty", ascending=False).head(n)