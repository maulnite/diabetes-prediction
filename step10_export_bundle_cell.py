from sklearn.base import clone
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import pandas as pd

# Pastikan variabel ini sudah ada dari step-step sebelumnya:
# df, ZERO_COLS, SEED, TOP_FEATS, best_model, best_thresh
df = pd.read_csv('./dataset/diabetes.csv')
X_raw = df.drop('Outcome', axis=1).copy()
y_full = df['Outcome'].copy()

X_raw[ZERO_COLS] = X_raw[ZERO_COLS].replace(0, np.nan)

export_imputer = IterativeImputer(random_state=SEED, max_iter=10)
X_imp = pd.DataFrame(
    export_imputer.fit_transform(X_raw),
    columns=X_raw.columns,
    index=X_raw.index,
)

winsor_bounds = {
    col: (float(X_imp[col].quantile(0.01)), float(X_imp[col].quantile(0.99)))
    for col in X_imp.columns
}

X_cap = X_imp.copy()
for col, (low, high) in winsor_bounds.items():
    X_cap[col] = X_cap[col].clip(low, high)

X_export = X_cap.copy()
X_export['Glucose_BMI'] = X_export['Glucose'] * X_export['BMI']
X_export['Glucose_Age'] = X_export['Glucose'] * X_export['Age']
X_export['Insulin_Glucose'] = X_export['Insulin'] / (X_export['Glucose'] + 1)
X_export['BMI_squared'] = X_export['BMI'] ** 2
X_export['BMI_cat'] = pd.cut(X_export['BMI'], bins=[0, 18.5, 25, 30, np.inf], labels=[0, 1, 2, 3], include_lowest=True).astype(int)
X_export['Age_group'] = pd.cut(X_export['Age'], bins=[0, 30, 45, 60, np.inf], labels=[0, 1, 2, 3], include_lowest=True).astype(int)
X_export = X_export[TOP_FEATS]

export_scaler = StandardScaler()
X_export_sc = export_scaler.fit_transform(X_export)

export_model = clone(best_model)
export_model.fit(X_export_sc, y_full)

train_probs = export_model.predict_proba(X_export_sc)[:, 1]
train_preds = (train_probs >= float(best_thresh)).astype(int)
train_scores_0_100 = np.round(train_probs * 100, 2)

def risk_text(prob, threshold):
    if prob >= max(0.8, threshold + 0.2):
        return 'Risiko tinggi'
    if prob >= threshold:
        return 'Risiko sedang–tinggi'
    if prob >= max(0.35, threshold - 0.1):
        return 'Risiko sedang'
    return 'Risiko rendah'

train_metrics = {
    'accuracy': float(accuracy_score(y_full, train_preds)),
    'precision': float(precision_score(y_full, train_preds, zero_division=0)),
    'recall': float(recall_score(y_full, train_preds, zero_division=0)),
    'f1': float(f1_score(y_full, train_preds, zero_division=0)),
    'roc_auc': float(roc_auc_score(y_full, train_probs)),
}

train_reference = {
    'scores_0_100': train_scores_0_100.tolist(),
    'probs': train_probs.tolist(),
    'y_true': y_full.astype(int).tolist(),
    'risk_bucket': [risk_text(p, float(best_thresh)) for p in train_probs],
    'n_samples': int(len(y_full)),
    'positive_rate': float(np.mean(y_full)),
    'metrics': train_metrics,
}

bundle = {
    'model': export_model,
    'scaler': export_scaler,
    'imputer': export_imputer,
    'winsor_bounds': winsor_bounds,
    'selected_features': TOP_FEATS,
    'zero_cols': ZERO_COLS,
    'threshold': float(best_thresh),
    'train_reference': train_reference,
}

joblib.dump(bundle, 'diabetes_model_bundle.pkl')
print('✅ Model bundle + train reference disimpan ke diabetes_model_bundle.pkl')
print('Train metrics:', train_metrics)
