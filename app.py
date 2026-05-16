import io
import json
import os
import pickle
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from flask import Flask, abort, flash, jsonify, redirect, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE_DIR / "model" / "diabetes_model_bundle_revised.pkl"))
OUTPUT_DIR = BASE_DIR / "data" / "predictions"
HISTORY_DIR = BASE_DIR / "data" / "history"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_PATH = HISTORY_DIR / "manual_prediction_history.csv"

ORIGINAL_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

FEATURE_GROUPS = [
    {
        "title": "Patient Basic Data",
        "icon": "bi-person-vcard",
        "description": "Data umum pasien yang dipakai model.",
        "features": ["Age", "Pregnancies"],
    },
    {
        "title": "Clinical Measurement",
        "icon": "bi-clipboard2-pulse",
        "description": "Pengukuran klinis utama yang paling mudah dibaca user.",
        "features": ["Glucose", "BloodPressure", "BMI"],
    },
    {
        "title": "Additional Medical Data",
        "icon": "bi-activity",
        "description": "Fitur tambahan yang tetap dibutuhkan karena model dilatih dengan format Pima Diabetes.",
        "features": ["SkinThickness", "Insulin", "DiabetesPedigreeFunction"],
    },
]

FIELD_META = {
    "Pregnancies": {
        "label": "Pregnancies", "help": "Jumlah kehamilan atau proxy delivery count", "step": "1", "placeholder": "2",
        "min": 0, "max": 20, "unit": "kali", "typical_min": 0, "typical_max": 15,
    },
    "Glucose": {
        "label": "Glucose", "help": "Kadar glukosa plasma", "step": "0.1", "placeholder": "120",
        "min": 0, "max": 300, "unit": "mg/dL", "typical_min": 50, "typical_max": 250,
    },
    "BloodPressure": {
        "label": "Blood Pressure", "help": "Tekanan darah diastolik", "step": "0.1", "placeholder": "72",
        "min": 0, "max": 200, "unit": "mmHg", "typical_min": 40, "typical_max": 140,
    },
    "SkinThickness": {
        "label": "Skin Thickness", "help": "Ketebalan lipatan kulit", "step": "0.1", "placeholder": "25",
        "min": 0, "max": 100, "unit": "mm", "typical_min": 5, "typical_max": 80,
    },
    "Insulin": {
        "label": "Insulin", "help": "Kadar insulin", "step": "0.1", "placeholder": "80",
        "min": 0, "max": 900, "unit": "mu U/ml", "typical_min": 5, "typical_max": 600,
    },
    "BMI": {
        "label": "BMI", "help": "Body Mass Index", "step": "0.1", "placeholder": "28.5",
        "min": 0, "max": 80, "unit": "kg/m²", "typical_min": 10, "typical_max": 70,
    },
    "DiabetesPedigreeFunction": {
        "label": "Diabetes Pedigree Function", "help": "Skor riwayat/genetik diabetes pada format Pima", "step": "0.001", "placeholder": "0.45",
        "min": 0, "max": 3, "unit": "score", "typical_min": 0, "typical_max": 2.5,
    },
    "Age": {
        "label": "Age", "help": "Usia pasien", "step": "1", "placeholder": "35",
        "min": 1, "max": 120, "unit": "tahun", "typical_min": 18, "typical_max": 90,
    },
}

SAMPLE_PATIENTS = {
    "low": {
        "label": "Low Risk Sample",
        "description": "Nilai relatif aman untuk demo prediksi risiko rendah.",
        "values": {
            "Pregnancies": 1, "Glucose": 92, "BloodPressure": 68, "SkinThickness": 22,
            "Insulin": 70, "BMI": 23.4, "DiabetesPedigreeFunction": 0.25, "Age": 28,
        },
    },
    "medium": {
        "label": "Medium Risk Sample",
        "description": "Kombinasi menengah untuk melihat kategori risiko sedang.",
        "values": {
            "Pregnancies": 3, "Glucose": 138, "BloodPressure": 82, "SkinThickness": 30,
            "Insulin": 135, "BMI": 31.2, "DiabetesPedigreeFunction": 0.62, "Age": 43,
        },
    },
    "high": {
        "label": "High Risk Sample",
        "description": "Kombinasi tinggi untuk demo prediksi diabetes/high risk.",
        "values": {
            "Pregnancies": 8, "Glucose": 190, "BloodPressure": 95, "SkinThickness": 40,
            "Insulin": 250, "BMI": 42, "DiabetesPedigreeFunction": 1.5, "Age": 60,
        },
    },
}

RISK_RULES = [
    ("Glucose", 140, "Kadar glucose tinggi", "Glucose berada di atas 140 sehingga menjadi sinyal risiko penting."),
    ("BMI", 30, "BMI tinggi", "BMI berada di atas 30 sehingga masuk area obesitas/berisiko."),
    ("Age", 45, "Usia lebih berisiko", "Usia di atas 45 membuat model memberi perhatian risiko lebih besar."),
    ("DiabetesPedigreeFunction", 0.5, "Riwayat/genetik cukup tinggi", "Nilai pedigree cukup tinggi sehingga faktor keturunan perlu diperhatikan."),
    ("BloodPressure", 80, "Tekanan darah cukup tinggi", "Blood pressure di atas 80 ikut menambah skor risiko pada rule aplikasi."),
]

REFERENCE_RULES = {
    "Glucose": [(70, "Low attention"), (140, "Reference range"), (10**9, "High attention")],
    "BMI": [(18.5, "Low BMI"), (25, "Reference range"), (30, "Overweight"), (10**9, "High BMI")],
    "BloodPressure": [(60, "Low attention"), (80, "Reference range"), (10**9, "Higher attention")],
    "Insulin": [(20, "Low attention"), (200, "Reference range"), (10**9, "Higher attention")],
    "SkinThickness": [(10, "Low attention"), (40, "Reference range"), (10**9, "Higher attention")],
    "DiabetesPedigreeFunction": [(0.5, "Reference range"), (10**9, "Higher family-history signal")],
    "Age": [(45, "Reference group"), (10**9, "Higher-risk age group")],
    "Pregnancies": [(5, "Reference group"), (10**9, "Higher pregnancy count")],
}

DATASET_MAPPING = [
    {
        "feature": "Pregnancies",
        "dataset_column": "Pregnancies",
        "role": "Input feature",
        "description": "Jumlah kehamilan yang tercatat pada dataset Pima.",
        "preprocessing": "Nilai 0 tetap valid karena dapat berarti belum pernah hamil.",
    },
    {
        "feature": "Glucose",
        "dataset_column": "Glucose",
        "role": "Input feature",
        "description": "Konsentrasi glukosa plasma 2 jam pada oral glucose tolerance test.",
        "preprocessing": "Nilai 0 diperlakukan sebagai missing value pada pipeline preprocessing.",
    },
    {
        "feature": "BloodPressure",
        "dataset_column": "BloodPressure",
        "role": "Input feature",
        "description": "Tekanan darah diastolik dalam mm Hg.",
        "preprocessing": "Nilai 0 diperlakukan sebagai missing value pada pipeline preprocessing.",
    },
    {
        "feature": "SkinThickness",
        "dataset_column": "SkinThickness",
        "role": "Input feature",
        "description": "Ketebalan lipatan kulit triceps dalam mm.",
        "preprocessing": "Nilai 0 diperlakukan sebagai missing value pada pipeline preprocessing.",
    },
    {
        "feature": "Insulin",
        "dataset_column": "Insulin",
        "role": "Input feature",
        "description": "Kadar serum insulin 2 jam dalam mu U/ml.",
        "preprocessing": "Nilai 0 diperlakukan sebagai missing value pada pipeline preprocessing.",
    },
    {
        "feature": "BMI",
        "dataset_column": "BMI",
        "role": "Input feature",
        "description": "Body Mass Index dengan satuan kg/m².",
        "preprocessing": "Nilai 0 diperlakukan sebagai missing value pada pipeline preprocessing.",
    },
    {
        "feature": "DiabetesPedigreeFunction",
        "dataset_column": "DiabetesPedigreeFunction",
        "role": "Input feature",
        "description": "Skor fungsi pedigree diabetes yang merepresentasikan riwayat/keturunan diabetes.",
        "preprocessing": "Dipakai langsung sebagai fitur numerik dan ikut dibuatkan feature engineering.",
    },
    {
        "feature": "Age",
        "dataset_column": "Age",
        "role": "Input feature",
        "description": "Usia pasien dalam tahun.",
        "preprocessing": "Dipakai langsung sebagai fitur numerik dan ikut dibuatkan feature engineering.",
    },
    {
        "feature": "Outcome",
        "dataset_column": "Outcome",
        "role": "Target label",
        "description": "Label target: 0 berarti non-diabetes, 1 berarti diabetes.",
        "preprocessing": "Tidak dipakai sebagai input saat prediksi; hanya dipakai saat training/evaluasi.",
    },
]

PIMA_DATASET_PATH = BASE_DIR / "data" / "diabetes.csv"
PIMA_DATASET_URL = "https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database"

REFERENCE_TABLE = [
    {"feature": "Glucose", "reference": "70 - 140 mg/dL", "attention": "> 140 mg/dL", "note": "Nilai glucose tinggi biasanya menjadi sinyal paling kuat untuk risiko diabetes."},
    {"feature": "BMI", "reference": "18.5 - 24.9 kg/m²", "attention": ">= 30 kg/m²", "note": "BMI tinggi/obesitas dapat meningkatkan risiko metabolik."},
    {"feature": "BloodPressure", "reference": "60 - 80 mmHg", "attention": "> 80 mmHg", "note": "Tekanan darah diastolik tinggi dapat menjadi faktor pendukung risiko."},
    {"feature": "Insulin", "reference": "20 - 200 mu U/ml", "attention": "> 200 mu U/ml", "note": "Nilai insulin perlu dibaca bersama glucose dan BMI."},
    {"feature": "SkinThickness", "reference": "10 - 40 mm", "attention": "> 40 mm", "note": "Fitur ini berasal dari format Pima dan ikut dipakai model."},
    {"feature": "DiabetesPedigreeFunction", "reference": "0 - 0.5", "attention": "> 0.5", "note": "Skor lebih tinggi menunjukkan sinyal riwayat/keturunan yang lebih kuat."},
    {"feature": "Age", "reference": "< 45 tahun", "attention": ">= 45 tahun", "note": "Usia lebih tua memberi sinyal risiko lebih besar pada rule aplikasi."},
    {"feature": "Pregnancies", "reference": "0 - 5 kali", "attention": "> 5 kali", "note": "Pregnancies tetap valid bernilai 0."},
]

GUIDE_STEPS = [
    {"title": "Manual Prediction", "icon": "bi-clipboard2-pulse", "text": "Isi 8 fitur medis utama, gunakan sample demo bila perlu, lalu klik Prediksi Sekarang."},
    {"title": "Review Result", "icon": "bi-search-heart", "text": "Baca label prediksi, probabilitas diabetes, kategori risiko, faktor risiko, dan normal range comparison."},
    {"title": "Batch CSV", "icon": "bi-file-earmark-spreadsheet", "text": "Upload file CSV dengan format Pima. Sistem akan memvalidasi kolom, mengimputasi missing value, dan melakukan clipping outlier."},
    {"title": "Download Report", "icon": "bi-download", "text": "Download hasil manual sebagai CSV/PDF atau hasil batch sebagai CSV untuk laporan."},
]

METHODOLOGY_STEPS = [
    {"step": "1", "title": "Dataset", "description": "Model dilatih menggunakan Pima Indians Diabetes Database dengan 8 fitur input numerik dan target Outcome."},
    {"step": "2", "title": "Missing Value Handling", "description": "Nilai 0 pada fitur medis tertentu diperlakukan sebagai missing value karena tidak realistis secara medis."},
    {"step": "3", "title": "Feature Engineering", "description": "Sistem membuat fitur interaksi seperti Glucose_BMI, Glucose_Age, BMI_Age, Risk_Score, dan fitur high-risk flag."},
    {"step": "4", "title": "Scaling", "description": "Fitur akhir diskalakan memakai scaler yang tersimpan di model bundle agar format input konsisten dengan training."},
    {"step": "5", "title": "Model Training", "description": "Beberapa model dibandingkan, lalu model terbaik berdasarkan performa digunakan untuk deployment."},
    {"step": "6", "title": "Evaluation", "description": "Performa dievaluasi dengan Accuracy, ROC-AUC, F1-Score, Precision, Recall, dan Confusion Matrix."},
    {"step": "7", "title": "Deployment", "description": "Model disimpan sebagai .pkl dan digunakan di aplikasi Flask + Bootstrap yang dapat dijalankan dengan Docker."},
]

MODEL_COMPARISON = [
    {"model": "Logistic Regression", "accuracy": 0.7532, "f1": 0.6481, "roc_auc": 0.8398, "precision": 0.6481, "recall": 0.6481},
    {"model": "Random Forest", "accuracy": 0.8636, "f1": 0.8037, "roc_auc": 0.9443, "precision": 0.8113, "recall": 0.7963},
    {"model": "Gradient Boosting", "accuracy": 0.8896, "f1": 0.8411, "roc_auc": 0.9563, "precision": 0.8491, "recall": 0.8333},
    {"model": "SVM", "accuracy": 0.8442, "f1": 0.7778, "roc_auc": 0.9115, "precision": 0.7778, "recall": 0.7778},
    {"model": "Stacking Ensemble", "accuracy": 0.8831, "f1": 0.8333, "roc_auc": 0.9480, "precision": 0.8333, "recall": 0.8333},
]

CONFUSION_MATRIX = {
    "labels": ["Non-Diabetes", "Diabetes"],
    "tn": 92,
    "fp": 8,
    "fn": 9,
    "tp": 45,
    "total": 154,
}

THRESHOLD_ANALYSIS = [
    {"threshold": 0.30, "accuracy": 0.8961, "f1": 0.8571, "recall": 0.8889, "precision": 0.8276},
    {"threshold": 0.35, "accuracy": 0.8831, "f1": 0.8364, "recall": 0.8519, "precision": 0.8214},
    {"threshold": 0.40, "accuracy": 0.8766, "f1": 0.8257, "recall": 0.8333, "precision": 0.8182},
    {"threshold": 0.45, "accuracy": 0.8896, "f1": 0.8411, "recall": 0.8333, "precision": 0.8491},
    {"threshold": 0.50, "accuracy": 0.8896, "f1": 0.8411, "recall": 0.8333, "precision": 0.8491},
    {"threshold": 0.55, "accuracy": 0.8961, "f1": 0.8491, "recall": 0.8333, "precision": 0.8654},
    {"threshold": 0.60, "accuracy": 0.8896, "f1": 0.8381, "recall": 0.8148, "precision": 0.8627},
    {"threshold": 0.65, "accuracy": 0.8896, "f1": 0.8317, "recall": 0.7778, "precision": 0.8936},
]


def get_pima_dataset_summary() -> Dict:
    summary = {
        "source_name": "Pima Indians Diabetes Database",
        "source_url": PIMA_DATASET_URL,
        "rows": "-",
        "columns": "-",
        "feature_count": len(ORIGINAL_FEATURES),
        "target_column": "Outcome",
        "non_diabetes": "-",
        "diabetes": "-",
        "input_columns": ORIGINAL_FEATURES,
    }
    try:
        if PIMA_DATASET_PATH.exists():
            df = pd.read_csv(PIMA_DATASET_PATH)
            summary["rows"] = int(df.shape[0])
            summary["columns"] = int(df.shape[1])
            if "Outcome" in df.columns:
                counts = df["Outcome"].value_counts().to_dict()
                summary["non_diabetes"] = int(counts.get(0, 0))
                summary["diabetes"] = int(counts.get(1, 0))
    except Exception:
        pass
    return summary

def load_bundle() -> Dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


bundle = load_bundle()
model = bundle["model"]
scaler = bundle["scaler"]
feature_cols: List[str] = bundle["feature_cols"]
zero_cols: List[str] = bundle.get("zero_cols", ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"])
metrics = bundle.get("metrics", {})
model_name = bundle.get("model_name", "Machine Learning Model")
TRAINING_MEAN = dict(zip(feature_cols, getattr(scaler, "mean_", np.zeros(len(feature_cols)))))


def format_percent(value: float) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "-"


def safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def risk_category(probability: float) -> Tuple[str, str, str]:
    if probability < 0.40:
        return "Low Risk", "success", "Risiko terindikasi diabetes relatif rendah berdasarkan model."
    if probability < 0.70:
        return "Medium Risk", "warning", "Risiko berada pada tingkat sedang, perlu perhatian pada faktor medis terkait."
    return "High Risk", "danger", "Risiko terindikasi diabetes tinggi berdasarkan model."


def validate_original_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in ORIGINAL_FEATURES if col not in df.columns]


def validate_manual_inputs(values: Dict[str, str]) -> Tuple[List[str], List[str], Dict[str, float]]:
    errors: List[str] = []
    warnings_list: List[str] = []
    numeric_values: Dict[str, float] = {}

    for feature in ORIGINAL_FEATURES:
        meta = FIELD_META[feature]
        raw_value = values.get(feature, "").strip()
        if raw_value == "":
            errors.append(f"{meta['label']} wajib diisi.")
            continue

        try:
            number = float(raw_value)
        except ValueError:
            errors.append(f"{meta['label']} harus berupa angka.")
            continue

        numeric_values[feature] = number
        if number < meta["min"] or number > meta["max"]:
            errors.append(f"{meta['label']} harus berada pada rentang {meta['min']} - {meta['max']} {meta['unit']}.")
            continue

        if feature in zero_cols and number == 0:
            warnings_list.append(f"{meta['label']} bernilai 0, sistem akan menganggapnya sebagai missing value dan mengisi otomatis.")
        elif number < meta["typical_min"] or number > meta["typical_max"]:
            warnings_list.append(f"{meta['label']} di luar rentang umum ({meta['typical_min']} - {meta['typical_max']} {meta['unit']}). Pastikan nilainya benar.")

    return errors, warnings_list, numeric_values


def validate_batch_values(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings_list: List[str] = []

    for col in ORIGINAL_FEATURES:
        raw_series = df[col]
        missing_mask = raw_series.isna() | raw_series.astype(str).str.strip().eq("")
        series = pd.to_numeric(raw_series, errors="coerce")
        non_numeric_mask = series.isna() & ~missing_mask
        invalid_count = int(non_numeric_mask.sum())
        if invalid_count:
            examples = raw_series[non_numeric_mask].astype(str).head(3).tolist()
            errors.append(f"Kolom {col} memiliki {invalid_count} nilai yang bukan angka. Contoh: {', '.join(examples)}")
            continue

        missing_count = int(missing_mask.sum())
        if missing_count:
            warnings_list.append(f"Kolom {col} memiliki {missing_count} nilai kosong. Nilai kosong akan diisi otomatis saat prediksi.")

        valid_series = series[~series.isna()]
        if valid_series.empty:
            warnings_list.append(f"Kolom {col} kosong semua. Sistem akan memakai nilai fallback dari data training.")
            continue

        meta = FIELD_META[col]
        out_of_range = int(((valid_series < meta["min"]) | (valid_series > meta["max"])).sum())
        if out_of_range:
            warnings_list.append(
                f"Kolom {col} memiliki {out_of_range} nilai di luar rentang valid ({meta['min']} - {meta['max']} {meta['unit']}). Nilai tersebut akan di-clip otomatis."
            )

    return errors, warnings_list


def build_data_quality_report(df: pd.DataFrame) -> Dict:
    columns = []
    total_missing = 0
    total_non_numeric = 0
    total_out_of_range = 0
    total_zero_as_missing = 0
    rows_with_issue = pd.Series(False, index=df.index)

    for col in ORIGINAL_FEATURES:
        raw_series = df[col]
        meta = FIELD_META[col]
        missing_mask = raw_series.isna() | raw_series.astype(str).str.strip().eq("")
        numeric = pd.to_numeric(raw_series, errors="coerce")
        non_numeric_mask = numeric.isna() & ~missing_mask
        valid_numeric = numeric[~numeric.isna()]
        out_low = int((valid_numeric < meta["min"]).sum())
        out_high = int((valid_numeric > meta["max"]).sum())
        zero_missing = int((numeric == 0).sum()) if col in zero_cols else 0
        missing = int(missing_mask.sum())
        non_numeric = int(non_numeric_mask.sum())
        out_of_range = out_low + out_high

        rows_with_issue = rows_with_issue | missing_mask | non_numeric_mask | (numeric < meta["min"]) | (numeric > meta["max"])
        total_missing += missing
        total_non_numeric += non_numeric
        total_out_of_range += out_of_range
        total_zero_as_missing += zero_missing

        status = "Clean"
        status_class = "success"
        if non_numeric:
            status = "Error"
            status_class = "danger"
        elif missing or out_of_range or zero_missing:
            status = "Adjusted"
            status_class = "warning"

        columns.append({
            "feature": col,
            "missing": missing,
            "non_numeric": non_numeric,
            "out_low": out_low,
            "out_high": out_high,
            "out_of_range": out_of_range,
            "zero_as_missing": zero_missing,
            "imputed": missing + zero_missing,
            "clipped": out_of_range,
            "status": status,
            "status_class": status_class,
        })

    total_cells = max(len(df) * len(ORIGINAL_FEATURES), 1)
    issue_score = 100 - round(((total_missing + total_non_numeric + total_out_of_range + total_zero_as_missing) / total_cells) * 100, 2)
    issue_score = max(issue_score, 0)

    return {
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "used_columns": len(ORIGINAL_FEATURES),
        "rows_with_issue": int(rows_with_issue.sum()),
        "total_missing": int(total_missing),
        "total_non_numeric": int(total_non_numeric),
        "total_out_of_range": int(total_out_of_range),
        "total_zero_as_missing": int(total_zero_as_missing),
        "quality_score": issue_score,
        "columns": columns,
    }


def clean_numeric_inputs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ORIGINAL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    existing_zero_cols = [col for col in zero_cols if col in df.columns]
    df[existing_zero_cols] = df[existing_zero_cols].replace(0, np.nan)

    for col in ORIGINAL_FEATURES:
        if df[col].isna().any():
            median_value = df[col].median()
            fallback_value = TRAINING_MEAN.get(col, 0)
            fill_value = median_value if pd.notna(median_value) else fallback_value
            df[col] = df[col].fillna(fill_value)

    for col in ORIGINAL_FEATURES:
        meta = FIELD_META[col]
        df[col] = df[col].clip(lower=meta["min"], upper=meta["max"])

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_numeric_inputs(df)

    safe_bmi = df["BMI"].replace(0, np.nan).fillna(TRAINING_MEAN.get("BMI", 1))
    safe_glucose = df["Glucose"].replace(0, np.nan).fillna(TRAINING_MEAN.get("Glucose", 1))

    df["Glucose_BMI"] = df["Glucose"] * df["BMI"]
    df["Glucose_Age"] = df["Glucose"] * df["Age"]
    df["BMI_Age"] = df["BMI"] * df["Age"]
    df["Insulin_Glucose"] = df["Insulin"] * df["Glucose"]
    df["Glucose_per_BMI"] = df["Glucose"] / safe_bmi
    df["Insulin_per_Glucose"] = df["Insulin"] / safe_glucose
    df["Age_per_Pregnancies"] = df["Age"] / (df["Pregnancies"] + 1)
    df["Glucose_squared"] = df["Glucose"] ** 2
    df["BMI_squared"] = df["BMI"] ** 2
    df["Age_squared"] = df["Age"] ** 2

    df["High_Glucose"] = (df["Glucose"] > 140).astype(int)
    df["High_BMI"] = (df["BMI"] > 30).astype(int)
    df["Older_Age"] = (df["Age"] > 45).astype(int)
    df["High_DPF"] = (df["DiabetesPedigreeFunction"] > 0.5).astype(int)
    df["High_BP"] = (df["BloodPressure"] > 80).astype(int)
    df["Risk_Score"] = df[["High_Glucose", "High_BMI", "Older_Age", "High_DPF", "High_BP"]].sum(axis=1)

    for col in feature_cols:
        if col not in df.columns:
            df[col] = TRAINING_MEAN.get(col, 0)

    return df


def detect_risk_factors(row: Dict) -> List[Dict[str, str]]:
    factors = []
    for feature, threshold, title, description in RISK_RULES:
        value = safe_float(row.get(feature, 0))
        if value > threshold:
            factors.append({
                "feature": feature,
                "title": title,
                "description": description,
                "value": round(value, 3),
                "threshold": threshold,
            })
    return factors


def detect_normal_factors(row: Dict) -> List[Dict[str, str]]:
    notes = []
    checks = [
        ("Glucose", 140, "Glucose belum melewati batas high-risk model"),
        ("BMI", 30, "BMI belum melewati batas high-risk model"),
        ("Age", 45, "Usia belum melewati batas high-risk model"),
        ("DiabetesPedigreeFunction", 0.5, "Pedigree/genetik belum tinggi menurut rule aplikasi"),
        ("BloodPressure", 80, "Blood pressure belum melewati batas high-risk model"),
    ]
    for feature, threshold, title in checks:
        value = safe_float(row.get(feature, 0))
        if value <= threshold:
            notes.append({"feature": feature, "title": title, "value": round(value, 3), "threshold": threshold})
    return notes


def normal_range_analysis(row: Dict) -> List[Dict[str, str]]:
    results = []
    for feature in ORIGINAL_FEATURES:
        value = safe_float(row.get(feature, 0))
        label = "Reference range"
        for upper, current_label in REFERENCE_RULES.get(feature, [(10**9, "Reference range")]):
            if value <= upper:
                label = current_label
                break
        if "High" in label or "Higher" in label or "Overweight" in label:
            status_class = "danger" if feature in ["Glucose", "BMI"] else "warning"
        elif "Low" in label:
            status_class = "secondary"
        else:
            status_class = "success"
        results.append({
            "feature": feature,
            "label": FIELD_META[feature]["label"],
            "value": value,
            "unit": FIELD_META[feature]["unit"],
            "status": label,
            "status_class": status_class,
        })
    return results


def build_interpretation(probability: float, risk_factors: List[Dict[str, str]]) -> str:
    category, _, _ = risk_category(probability)
    if not risk_factors:
        return "Tidak ada faktor risiko utama yang melewati ambang sederhana aplikasi. Model tetap melihat kombinasi fitur lengkap, jadi hasil akhir tetap mengikuti probabilitas model."

    names = ", ".join([item["feature"] for item in risk_factors[:3]])
    if category == "High Risk":
        return f"Model memberi risiko tinggi. Faktor yang paling terlihat dari input adalah {names}."
    if category == "Medium Risk":
        return f"Model memberi risiko sedang. Beberapa faktor yang perlu diperhatikan adalah {names}."
    return f"Model memberi risiko rendah, tetapi ada faktor yang tetap perlu dicek: {names}."


def predict_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    engineered = engineer_features(df)
    X = scaler.transform(engineered[feature_cols])
    predictions = model.predict(X)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]
    else:
        probabilities = predictions.astype(float)

    result = clean_numeric_inputs(df.copy())
    result["Prediction"] = predictions.astype(int)
    result["Probability_Diabetes"] = np.round(probabilities, 4)
    result["Probability_Percent"] = (probabilities * 100).round(2)
    result["Risk_Category"] = [risk_category(float(prob))[0] for prob in probabilities]
    result["Label"] = np.where(result["Prediction"] == 1, "Diabetes", "Non-Diabetes")
    result["Risk_Factors"] = [", ".join([item["feature"] for item in detect_risk_factors(row)]) or "-" for row in result.to_dict(orient="records")]
    return result


def get_top_features(limit: int = 8):
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []
    pairs = sorted(zip(feature_cols, importances), key=lambda item: item[1], reverse=True)
    return [(name, round(float(score) * 100, 2)) for name, score in pairs[:limit]]


def get_dashboard_stats() -> Dict:
    records, _ = read_history(limit=1000)
    if not records:
        return {"total": 0, "high": 0, "medium": 0, "low": 0, "diabetes": 0, "recent": []}
    df = pd.DataFrame(records)
    return {
        "total": int(len(df)),
        "high": int((df.get("Risk_Category") == "High Risk").sum()),
        "medium": int((df.get("Risk_Category") == "Medium Risk").sum()),
        "low": int((df.get("Risk_Category") == "Low Risk").sum()),
        "diabetes": int((df.get("Label") == "Diabetes").sum()),
        "recent": records[:5],
    }


def save_manual_result(input_values: Dict[str, float], result_row: Dict, risk_factors: List[Dict], interpretation: str) -> str:
    file_id = f"manual_prediction_{uuid.uuid4().hex}.csv"
    output_path = OUTPUT_DIR / file_id

    probability = float(result_row["Probability_Diabetes"])
    category, _, _ = risk_category(probability)
    saved_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record = {
        "SavedAt": saved_at,
        **input_values,
        "Prediction": int(result_row["Prediction"]),
        "Label": result_row["Label"],
        "Probability_Diabetes": round(probability, 4),
        "Probability_Percent": round(probability * 100, 2),
        "Risk_Category": category,
        "Risk_Factors": ", ".join([item["feature"] for item in risk_factors]) or "-",
        "Interpretation": interpretation,
    }
    pd.DataFrame([record]).to_csv(output_path, index=False)

    history_df = pd.DataFrame([record])
    if HISTORY_PATH.exists():
        history_df.to_csv(HISTORY_PATH, mode="a", header=False, index=False)
    else:
        history_df.to_csv(HISTORY_PATH, index=False)

    return file_id


def read_history(limit: int = 100) -> Tuple[List[Dict], List[str]]:
    if not HISTORY_PATH.exists():
        return [], []
    try:
        df = pd.read_csv(HISTORY_PATH)
    except Exception:
        return [], []
    df = df.tail(limit).iloc[::-1]
    return df.to_dict(orient="records"), df.columns.tolist()


def create_pdf_report(record: Dict) -> io.BytesIO:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=2 * cm, leftMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Diabetes Prediction Report", styles["Title"]))
    story.append(Paragraph("Machine Learning based diabetes risk prediction result", styles["Normal"]))
    story.append(Spacer(1, 0.5 * cm))

    summary = [
        ["Prediction", str(record.get("Label", "-"))],
        ["Risk Category", str(record.get("Risk_Category", "-"))],
        ["Probability", f"{safe_float(record.get('Probability_Percent')):.2f}%"],
        ["Saved At", str(record.get("SavedAt", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))],
    ]
    summary_table = Table(summary, hAlign="LEFT", colWidths=[5 * cm, 9 * cm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8f1ff")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#0f172a")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dbe5f2")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.6 * cm))

    story.append(Paragraph("Patient Input", styles["Heading2"]))
    data = [["Feature", "Value"]]
    for feature in ORIGINAL_FEATURES:
        data.append([feature, str(record.get(feature, "-"))])
    input_table = Table(data, hAlign="LEFT", colWidths=[7 * cm, 7 * cm])
    input_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563eb")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dbe5f2")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("PADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(input_table)
    story.append(Spacer(1, 0.6 * cm))

    story.append(Paragraph("Interpretation", styles["Heading2"]))
    story.append(Paragraph(str(record.get("Interpretation", "-")), styles["Normal"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("Disclaimer: this report is for learning and demonstration only. It is not an official medical diagnosis.", styles["Italic"]))

    doc.build(story)
    buffer.seek(0)
    return buffer


app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-this")
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024


@app.context_processor
def inject_global_data():
    return {
        "model_name": model_name,
        "metrics": metrics,
        "format_percent": format_percent,
        "current_year": datetime.now().year,
    }


@app.route("/")
def index():
    return render_template("index.html", top_features=get_top_features(5), dashboard_stats=get_dashboard_stats())


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html", fields=FIELD_META, values={}, samples=SAMPLE_PATIENTS, feature_groups=FEATURE_GROUPS, reference_table=REFERENCE_TABLE)

    values = {feature: request.form.get(feature, "").strip() for feature in ORIGINAL_FEATURES}
    errors, warnings_list, numeric_values = validate_manual_inputs(values)

    if errors:
        for error in errors:
            flash(error, "danger")
        for warning in warnings_list:
            flash(warning, "warning")
        return render_template("predict.html", fields=FIELD_META, values=values, samples=SAMPLE_PATIENTS, feature_groups=FEATURE_GROUPS, reference_table=REFERENCE_TABLE), 400

    for warning in warnings_list:
        flash(warning, "warning")

    input_df = pd.DataFrame([numeric_values])
    result_df = predict_dataframe(input_df)
    row = result_df.iloc[0].to_dict()
    probability = float(row["Probability_Diabetes"])
    category, badge_class, explanation = risk_category(probability)
    risk_factors = detect_risk_factors(numeric_values)
    normal_factors = detect_normal_factors(numeric_values)
    comparison = normal_range_analysis(numeric_values)
    interpretation = build_interpretation(probability, risk_factors)
    file_id = save_manual_result(numeric_values, row, risk_factors, interpretation)

    return render_template(
        "result.html",
        input_values=input_df.iloc[0].to_dict(),
        result=row,
        probability_percent=round(probability * 100, 2),
        category=category,
        badge_class=badge_class,
        explanation=explanation,
        risk_factors=risk_factors,
        normal_factors=normal_factors,
        comparison=comparison,
        interpretation=interpretation,
        file_id=file_id,
    )


@app.route("/batch", methods=["GET", "POST"])
def batch():
    if request.method == "GET":
        return render_template("batch.html", required_columns=ORIGINAL_FEATURES)

    file = request.files.get("csv_file")
    if not file or file.filename == "":
        flash("File CSV belum dipilih.", "danger")
        return redirect(url_for("batch"))

    filename = secure_filename(file.filename)
    if not filename.lower().endswith(".csv"):
        flash("Format file harus .csv.", "danger")
        return redirect(url_for("batch"))

    try:
        df = pd.read_csv(file)
    except Exception as exc:
        flash(f"CSV gagal dibaca: {exc}", "danger")
        return redirect(url_for("batch"))

    missing_cols = validate_original_columns(df)
    if missing_cols:
        flash(f"Kolom CSV belum lengkap: {', '.join(missing_cols)}", "danger")
        return render_template("batch.html", required_columns=ORIGINAL_FEATURES), 400

    quality_report = build_data_quality_report(df[ORIGINAL_FEATURES].copy())
    errors, warnings_list = validate_batch_values(df[ORIGINAL_FEATURES].copy())
    if errors:
        for error in errors[:8]:
            flash(error, "danger")
        if len(errors) > 8:
            flash(f"Ada {len(errors) - 8} error lain di CSV.", "danger")
        return render_template("batch.html", required_columns=ORIGINAL_FEATURES), 400

    for warning in warnings_list[:6]:
        flash(warning, "warning")

    try:
        result_df = predict_dataframe(df[ORIGINAL_FEATURES].copy())
    except Exception as exc:
        flash(f"Prediksi gagal diproses: {exc}", "danger")
        return render_template("batch.html", required_columns=ORIGINAL_FEATURES), 500

    diabetes_count = int((result_df["Prediction"] == 1).sum())
    total_rows = len(result_df)
    category_counts = result_df["Risk_Category"].value_counts().to_dict()
    file_id = f"prediction_{uuid.uuid4().hex}.csv"
    output_path = OUTPUT_DIR / file_id
    result_df.to_csv(output_path, index=False)

    preview_rows = result_df.head(100).to_dict(orient="records")
    preview_columns = result_df.columns.tolist()
    shown_rows = len(preview_rows)
    chart_data = {
        "labels": ["Low Risk", "Medium Risk", "High Risk"],
        "values": [int(category_counts.get("Low Risk", 0)), int(category_counts.get("Medium Risk", 0)), int(category_counts.get("High Risk", 0))],
    }
    prediction_chart_data = {
        "labels": ["Non-Diabetes", "Diabetes"],
        "values": [total_rows - diabetes_count, diabetes_count],
    }

    return render_template(
        "batch_result.html",
        filename=filename,
        total_rows=total_rows,
        diabetes_count=diabetes_count,
        non_diabetes_count=total_rows - diabetes_count,
        low_risk_count=int(category_counts.get("Low Risk", 0)),
        medium_risk_count=int(category_counts.get("Medium Risk", 0)),
        high_risk_count=int(category_counts.get("High Risk", 0)),
        preview_rows=preview_rows,
        preview_columns=preview_columns,
        file_id=file_id,
        quality_report=quality_report,
        chart_data=chart_data,
        prediction_chart_data=prediction_chart_data,
        shown_rows=shown_rows,
    )


@app.route("/history")
def history():
    records, columns = read_history()
    return render_template("history.html", records=records, columns=columns)


@app.route("/history/download")
def download_history():
    if not HISTORY_PATH.exists():
        flash("Belum ada history prediksi manual untuk di-download.", "warning")
        return redirect(url_for("history"))
    return send_file(HISTORY_PATH, mimetype="text/csv", as_attachment=True, download_name="manual_prediction_history.csv")


@app.route("/history/clear", methods=["POST"])
def clear_history():
    if HISTORY_PATH.exists():
        HISTORY_PATH.unlink()
    flash("History prediksi manual berhasil dihapus.", "success")
    return redirect(url_for("history"))


@app.route("/download/<path:file_id>")
def download(file_id):
    safe_file = secure_filename(file_id)
    output_path = OUTPUT_DIR / safe_file
    if not output_path.exists():
        abort(404)
    download_name = "hasil_prediksi_diabetes.csv" if safe_file.startswith("prediction_") else "hasil_prediksi_manual.csv"
    return send_file(output_path, mimetype="text/csv", as_attachment=True, download_name=download_name)


@app.route("/report/<path:file_id>")
def report_pdf(file_id):
    safe_file = secure_filename(file_id)
    output_path = OUTPUT_DIR / safe_file
    if not output_path.exists() or not safe_file.startswith("manual_prediction_"):
        abort(404)
    try:
        record = pd.read_csv(output_path).iloc[0].to_dict()
        pdf_buffer = create_pdf_report(record)
    except Exception as exc:
        flash(f"PDF report gagal dibuat: {exc}", "danger")
        return redirect(url_for("history"))
    return send_file(pdf_buffer, mimetype="application/pdf", as_attachment=True, download_name="diabetes_prediction_report.pdf")


@app.route("/guide")
def prediction_guide():
    return render_template("guide.html", guide_steps=GUIDE_STEPS, reference_table=REFERENCE_TABLE, required_columns=ORIGINAL_FEATURES)


@app.route("/methodology")
def methodology():
    return render_template("methodology.html", methodology_steps=METHODOLOGY_STEPS, feature_cols=feature_cols, zero_cols=zero_cols, original_features=ORIGINAL_FEATURES)


@app.route("/batch/template")
def download_csv_template():
    template_df = pd.DataFrame([
        {"Pregnancies": 2, "Glucose": 120, "BloodPressure": 72, "SkinThickness": 25, "Insulin": 80, "BMI": 28.5, "DiabetesPedigreeFunction": 0.45, "Age": 35},
        {"Pregnancies": 8, "Glucose": 190, "BloodPressure": 95, "SkinThickness": 40, "Insulin": 250, "BMI": 42.0, "DiabetesPedigreeFunction": 1.50, "Age": 60},
    ])
    csv_text = template_df.to_csv(index=False)
    buffer = io.BytesIO(csv_text.encode("utf-8"))
    buffer.seek(0)
    return send_file(buffer, mimetype="text/csv", as_attachment=True, download_name="diabetes_batch_template.csv")


@app.route("/about")
def about():
    return render_template(
        "about.html",
        feature_cols=feature_cols,
        zero_cols=zero_cols,
        original_features=ORIGINAL_FEATURES,
        top_features=get_top_features(10),
        risk_rules=RISK_RULES,
        fields=FIELD_META,
    )


@app.route("/dataset-source")
def dataset_source():
    return render_template("dataset_source.html", mapping=DATASET_MAPPING, original_features=ORIGINAL_FEATURES, pima_summary=get_pima_dataset_summary())


@app.route("/model-performance")
def model_performance():
    top_features = get_top_features(15)
    feature_chart = {
        "labels": [name for name, _ in top_features],
        "values": [score for _, score in top_features],
    }
    threshold_chart = {
        "labels": [item["threshold"] for item in THRESHOLD_ANALYSIS],
        "f1": [item["f1"] for item in THRESHOLD_ANALYSIS],
        "recall": [item["recall"] for item in THRESHOLD_ANALYSIS],
        "precision": [item["precision"] for item in THRESHOLD_ANALYSIS],
    }
    return render_template(
        "model_performance.html",
        top_features=top_features,
        feature_cols=feature_cols,
        metrics=metrics,
        confusion_matrix=CONFUSION_MATRIX,
        model_comparison=MODEL_COMPARISON,
        threshold_analysis=THRESHOLD_ANALYSIS,
        feature_chart=feature_chart,
        threshold_chart=threshold_chart,
    )


@app.route("/api-docs")
def api_docs():
    return render_template("api_docs.html", sample=SAMPLE_PATIENTS["high"]["values"], fields=FIELD_META)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or {}
    values = {feature: str(payload.get(feature, "")).strip() for feature in ORIGINAL_FEATURES}
    errors, warnings_list, numeric_values = validate_manual_inputs(values)
    if errors:
        return jsonify({"ok": False, "errors": errors, "warnings": warnings_list}), 400

    result_df = predict_dataframe(pd.DataFrame([numeric_values]))
    row = result_df.iloc[0].to_dict()
    probability = float(row["Probability_Diabetes"])
    category, _, explanation = risk_category(probability)
    risk_factors = detect_risk_factors(numeric_values)
    interpretation = build_interpretation(probability, risk_factors)
    return jsonify({
        "ok": True,
        "prediction": row["Label"],
        "prediction_code": int(row["Prediction"]),
        "probability_diabetes": round(probability, 4),
        "probability_percent": round(probability * 100, 2),
        "risk_category": category,
        "explanation": explanation,
        "risk_factors": risk_factors,
        "interpretation": interpretation,
        "warnings": warnings_list,
    })


@app.errorhandler(404)
def not_found(_):
    return render_template("error.html", code=404, title="Halaman tidak ditemukan", message="URL yang kamu buka tidak tersedia."), 404


@app.errorhandler(500)
def server_error(_):
    return render_template("error.html", code=500, title="Terjadi kesalahan server", message="Coba ulangi proses atau cek format input yang digunakan."), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
