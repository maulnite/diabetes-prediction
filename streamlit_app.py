from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

st.set_page_config(
    page_title="Diabetes Risk Predictor Pro+",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

RAW_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

RISK_ORDER = ["Risiko rendah", "Risiko sedang", "Risiko sedang–tinggi", "Risiko tinggi"]
LABEL_CANDIDATES = ["Outcome", "Label", "target", "Target", "y_true", "actual", "Actual"]


@st.cache_resource
def load_bundle():
    candidates = [Path("./model/diabetes_model_bundle_revised.pkl")]
    for path in candidates:
        if path.exists():
            obj = joblib.load(path)
            required = {"model", "scaler", "imputer", "selected_features", "threshold"}
            if isinstance(obj, dict) and required.issubset(obj.keys()):
                return obj, str(path)
            raise ValueError(
                "File model ditemukan, tetapi formatnya bukan bundle lengkap. "
                "Gunakan bundle yang berisi model, imputer, scaler, dan threshold."
            )
    raise FileNotFoundError(
        "File 'diabetes_model_bundle.pkl' belum ditemukan di folder yang sama dengan app."
    )


def apply_custom_css():
    st.markdown(
        """
        <style>
        .main {
            background: linear-gradient(180deg, #f8fbff 0%, #eef5ff 100%);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }
        .hero-card {
            background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%);
            padding: 1.35rem 1.5rem;
            border-radius: 24px;
            color: white;
            box-shadow: 0 10px 28px rgba(29, 78, 216, 0.18);
            margin-bottom: 1rem;
        }
        .sub-card {
            background: white;
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 20px;
            padding: 1rem 1.1rem;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
            margin-bottom: 1rem;
        }
        .metric-card {
            background: white;
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 18px;
            padding: 1rem;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
        }
        .risk-pill {
            display: inline-block;
            padding: 0.45rem 0.85rem;
            border-radius: 999px;
            font-size: 0.95rem;
            font-weight: 700;
            margin-top: 0.2rem;
        }
        .risk-low { background: #dcfce7; color: #166534; }
        .risk-medium { background: #fef3c7; color: #92400e; }
        .risk-mid-high { background: #fee2e2; color: #b91c1c; }
        .risk-high { background: #fecaca; color: #991b1b; }
        .small-note {
            color: #475569;
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(title: str, value: str, caption: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div style="font-size:0.92rem;color:#475569;">{title}</div>
            <div style="font-size:1.8rem;font-weight:800;color:#0f172a;line-height:1.25;">{value}</div>
            <div style="font-size:0.86rem;color:#64748b;">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sanitize_input_df(df: pd.DataFrame):
    clean = df.copy()
    for col in RAW_COLUMNS:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    invalid_rows = clean[RAW_COLUMNS].isna().any(axis=1)
    return clean, invalid_rows


def preprocess_input(df_input: pd.DataFrame, bundle: dict):
    zero_cols = bundle.get("zero_cols", ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"])
    imputer = bundle["imputer"]
    winsor_bounds = bundle.get("winsor_bounds", {})
    selected_features = bundle["selected_features"]
    scaler = bundle["scaler"]

    df = df_input.copy()[RAW_COLUMNS]
    df[zero_cols] = df[zero_cols].replace(0, np.nan)

    df = pd.DataFrame(imputer.transform(df), columns=df.columns, index=df.index)

    for col, bounds in winsor_bounds.items():
        if col in df.columns:
            low, high = bounds
            df[col] = df[col].clip(low, high)

    df["Glucose_BMI"] = df["Glucose"] * df["BMI"]
    df["Glucose_Age"] = df["Glucose"] * df["Age"]
    df["Insulin_Glucose"] = df["Insulin"] / (df["Glucose"] + 1)
    df["BMI_squared"] = df["BMI"] ** 2

    df["BMI_cat"] = pd.cut(
        df["BMI"],
        bins=[0, 18.5, 25, 30, np.inf],
        labels=[0, 1, 2, 3],
        include_lowest=True,
    ).astype(int)

    df["Age_group"] = pd.cut(
        df["Age"],
        bins=[0, 30, 45, 60, np.inf],
        labels=[0, 1, 2, 3],
        include_lowest=True,
    ).astype(int)

    df_selected = df[selected_features].copy()
    df_scaled = scaler.transform(df_selected)
    return df, df_selected, df_scaled


def risk_text(prob: float, threshold: float):
    if prob >= max(0.8, threshold + 0.2):
        return "Risiko tinggi"
    if prob >= threshold:
        return "Risiko sedang–tinggi"
    if prob >= max(0.35, threshold - 0.1):
        return "Risiko sedang"
    return "Risiko rendah"


def risk_css_class(risk_label: str):
    mapping = {
        "Risiko rendah": "risk-low",
        "Risiko sedang": "risk-medium",
        "Risiko sedang–tinggi": "risk-mid-high",
        "Risiko tinggi": "risk-high",
    }
    return mapping.get(risk_label, "risk-medium")


def predict(df_input: pd.DataFrame, bundle: dict, passthrough_df: pd.DataFrame | None = None):
    df_processed, df_selected, X_ready = preprocess_input(df_input, bundle)
    probs = bundle["model"].predict_proba(X_ready)[:, 1]
    threshold = float(bundle.get("threshold", 0.5))
    preds = (probs >= threshold).astype(int)

    result = (passthrough_df.copy() if passthrough_df is not None else df_input.copy()).reset_index(drop=True)
    result["probability_diabetes"] = probs
    result["model_score_0_100"] = np.round(probs * 100, 2)
    result["confidence_score_0_100"] = np.round(np.maximum(probs, 1 - probs) * 100, 2)
    result["risk_bucket"] = [risk_text(p, threshold) for p in probs]
    result["prediction_label"] = np.where(preds == 1, "Diabetes", "Tidak Diabetes")
    result["prediction_binary"] = preds
    return result, df_processed, df_selected, threshold


def validate_input_df(df: pd.DataFrame):
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in RAW_COLUMNS]
    return missing, extra


def make_template_df():
    return pd.DataFrame(
        [
            {
                "Pregnancies": 2,
                "Glucose": 130,
                "BloodPressure": 80,
                "SkinThickness": 25,
                "Insulin": 80,
                "BMI": 28.0,
                "DiabetesPedigreeFunction": 0.50,
                "Age": 40,
            },
            {
                "Pregnancies": 5,
                "Glucose": 168,
                "BloodPressure": 74,
                "SkinThickness": 0,
                "Insulin": 0,
                "BMI": 34.6,
                "DiabetesPedigreeFunction": 0.63,
                "Age": 51,
            },
        ]
    )


def find_label_column(df: pd.DataFrame):
    for col in LABEL_CANDIDATES:
        if col in df.columns:
            return col
    return None


def compute_metrics_from_probs(y_true: pd.Series, probs: np.ndarray, threshold: float):
    y_num = pd.to_numeric(y_true, errors="coerce")
    mask = y_num.isin([0, 1])
    if mask.sum() == 0:
        return None, 0

    y = y_num[mask].astype(int).to_numpy()
    p = np.asarray(probs)[mask.to_numpy()]
    preds = (p >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
    }
    if len(np.unique(y)) >= 2:
        metrics["roc_auc"] = float(roc_auc_score(y, p))
    else:
        metrics["roc_auc"] = np.nan
    return metrics, int(mask.sum())


def distribution_from_scores(scores, as_percent=True):
    labels = [f"{i}-{i+10}" for i in range(0, 100, 10)]
    bins = np.linspace(0, 100, 11)
    s = pd.Series(scores, dtype=float).clip(0, 100)
    cats = pd.cut(s, bins=bins, labels=labels, include_lowest=True, right=True)
    dist = cats.value_counts(sort=False).reindex(labels, fill_value=0).astype(float)
    if as_percent and dist.sum() > 0:
        dist = dist / dist.sum() * 100
    return dist


def risk_distribution(series, as_percent=True):
    dist = pd.Series(series).value_counts().reindex(RISK_ORDER, fill_value=0).astype(float)
    if as_percent and dist.sum() > 0:
        dist = dist / dist.sum() * 100
    return dist


def extract_train_reference(bundle: dict, threshold: float):
    ref = bundle.get("train_reference")
    if not isinstance(ref, dict):
        return None

    scores = ref.get("scores_0_100")
    if scores is None:
        probs = ref.get("probs")
        if probs is not None:
            scores = np.asarray(probs, dtype=float) * 100

    y_true = ref.get("y_true")
    metrics = ref.get("metrics")
    if metrics is None and y_true is not None and scores is not None:
        metrics, _ = compute_metrics_from_probs(pd.Series(y_true), np.asarray(scores) / 100.0, threshold)

    return {
        "scores_0_100": np.asarray(scores, dtype=float) if scores is not None else None,
        "risk_bucket": ref.get("risk_bucket"),
        "metrics": metrics,
        "n_samples": ref.get("n_samples"),
        "positive_rate": ref.get("positive_rate"),
        "y_true": np.asarray(y_true) if y_true is not None else None,
    }


apply_custom_css()

try:
    bundle, bundle_path = load_bundle()
except Exception as exc:
    st.error(str(exc))
    st.stop()

threshold = float(bundle.get("threshold", 0.5))
selected_features = bundle.get("selected_features", [])
train_reference = extract_train_reference(bundle, threshold)

st.markdown(
    f"""
    <div class="hero-card">
        <div style="font-size:0.95rem; opacity:0.9;">Clinical ML Demo</div>
        <div style="font-size:2rem; font-weight:800; margin-top:0.15rem;">🩺 Diabetes Risk Predictor Pro+</div>
        <div style="margin-top:0.45rem; font-size:1rem; opacity:0.95; max-width:980px;">
            App ini menerima data mentah pasien, menjalankan imputasi, winsor/capping, feature engineering,
            scaling, lalu memprediksi probabilitas diabetes menggunakan bundle model dari notebook.
            Versi ini juga bisa menampilkan grafik pembanding train-fit reference vs hasil prediksi batch.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

hero_col1, hero_col2, hero_col3, hero_col4 = st.columns(4)
with hero_col1:
    metric_card("Threshold", f"{threshold:.2f}", "Batas keputusan model")
with hero_col2:
    metric_card("Fitur akhir", str(len(selected_features)), "Jumlah fitur yang dipakai model")
with hero_col3:
    metric_card("Bundle", "Loaded", Path(bundle_path).name)
with hero_col4:
    metric_card("Train reference", "Ready" if train_reference else "Not found", "Untuk grafik pembanding")

with st.sidebar:
    st.markdown("### Konfigurasi Model")
    st.success(f"Bundle aktif: {Path(bundle_path).name}")
    st.write(f"**Threshold keputusan:** {threshold:.2f}")
    st.markdown("**Fitur akhir yang dipakai model**")
    st.code("\n".join(map(str, selected_features)) if selected_features else "-", language="text")
    st.info(
        "Nilai 0 pada Glucose, BloodPressure, SkinThickness, Insulin, dan BMI diperlakukan sebagai missing lalu diimputasi."
    )
    if train_reference:
        st.success("Train reference ditemukan. Grafik pembanding train vs batch aktif.")
    else:
        st.warning(
            "Train reference belum ada di bundle. Prediksi tetap jalan, tapi grafik pembanding train vs batch belum bisa ditampilkan."
        )
    st.markdown("---")
    st.caption(
        "Catatan: train reference di sini adalah hasil dari data training yang disimpan ke bundle. Ini berguna untuk pembanding distribusi skor, "
        "tetapi bukan pengganti evaluasi validation/test."
    )

single_tab, batch_tab = st.tabs(["Prediksi 1 Pasien", "Prediksi Batch CSV"])

with single_tab:
    form_col, info_col = st.columns([1.25, 1])

    with form_col:
        st.markdown('<div class="sub-card">', unsafe_allow_html=True)
        st.subheader("Input data pasien")
        with st.form("single_prediction_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2, step=1)
                glucose = st.number_input("Glucose", min_value=0.0, max_value=300.0, value=130.0, step=1.0)
                blood_pressure = st.number_input("BloodPressure", min_value=0.0, max_value=200.0, value=80.0, step=1.0)
            with c2:
                skin_thickness = st.number_input("SkinThickness", min_value=0.0, max_value=120.0, value=25.0, step=1.0)
                insulin = st.number_input("Insulin", min_value=0.0, max_value=1000.0, value=80.0, step=1.0)
                bmi = st.number_input("BMI", min_value=0.0, max_value=80.0, value=28.0, step=0.1)
            with c3:
                dpf = st.number_input("DiabetesPedigreeFunction", min_value=0.0, max_value=3.0, value=0.50, step=0.01)
                age = st.number_input("Age", min_value=1, max_value=120, value=40, step=1)
            submitted = st.form_submit_button("Prediksi sekarang", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with info_col:
        st.markdown('<div class="sub-card">', unsafe_allow_html=True)
        st.subheader("Tentang skor")
        st.markdown(
            """
            <div class="small-note">
            <b>model_score_0_100</b> = probabilitas diabetes dalam skala 0–100.<br>
            <b>confidence_score_0_100</b> = seberapa jauh model condong ke salah satu kelas.<br>
            <b>risk_bucket</b> = kategori risiko berdasarkan threshold model.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    if submitted:
        input_df = pd.DataFrame(
            [
                {
                    "Pregnancies": pregnancies,
                    "Glucose": glucose,
                    "BloodPressure": blood_pressure,
                    "SkinThickness": skin_thickness,
                    "Insulin": insulin,
                    "BMI": bmi,
                    "DiabetesPedigreeFunction": dpf,
                    "Age": age,
                }
            ]
        )

        result_df, processed_df, selected_df, used_threshold = predict(input_df, bundle)
        prob = float(result_df.loc[0, "probability_diabetes"])
        score = float(result_df.loc[0, "model_score_0_100"])
        confidence = float(result_df.loc[0, "confidence_score_0_100"])
        label = result_df.loc[0, "prediction_label"]
        risk_level = result_df.loc[0, "risk_bucket"]
        risk_class = risk_css_class(risk_level)

        st.markdown("### Hasil prediksi")
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            metric_card("Prediksi", label, "Kelas akhir model")
        with r2:
            metric_card("Risk Score", f"{score:.2f}", "Skala 0–100")
        with r3:
            metric_card("Probability", f"{prob:.2%}", "Probabilitas kelas diabetes")
        with r4:
            metric_card("Confidence", f"{confidence:.2f}", "Semakin tinggi, semakin yakin")

        st.markdown(f'<div class="risk-pill {risk_class}">{risk_level}</div>', unsafe_allow_html=True)
        st.progress(float(np.clip(prob, 0.0, 1.0)))
        st.caption(f"Threshold keputusan yang dipakai: {used_threshold:.2f}")

        with st.expander("Lihat data setelah preprocessing"):
            st.dataframe(processed_df, use_container_width=True, hide_index=True)
        with st.expander("Lihat fitur akhir yang masuk ke model"):
            st.dataframe(selected_df, use_container_width=True, hide_index=True)

with batch_tab:
    top_col, helper_col = st.columns([1.2, 1])
    with top_col:
        st.markdown('<div class="sub-card">', unsafe_allow_html=True)
        st.subheader("Upload CSV untuk prediksi batch")
        st.caption("CSV wajib memiliki kolom mentah yang sama seperti saat training. Kolom label opsional: Outcome / Label / target.")
        st.code(", ".join(RAW_COLUMNS), language="text")
        template_df = make_template_df()
        st.download_button(
            "Download template CSV",
            data=template_df.to_csv(index=False).encode("utf-8"),
            file_name="template_diabetes_input.csv",
            mime="text/csv",
            use_container_width=True,
        )
        uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
        st.markdown('</div>', unsafe_allow_html=True)

    with helper_col:
        st.markdown('<div class="sub-card">', unsafe_allow_html=True)
        st.subheader("Grafik pembanding")
        st.markdown(
            """
            <div class="small-note">
            Kalau bundle menyimpan <b>train_reference</b>, app akan menampilkan:<br>
            • distribusi model score train vs batch<br>
            • distribusi risk bucket train vs batch<br>
            • metrik train-fit vs batch, bila file batch punya label asli<br><br>
            Kolom label yang dikenali: <b>Outcome</b>, <b>Label</b>, <b>target</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        batch_original = pd.read_csv(uploaded_file)
        label_col = find_label_column(batch_original)
        batch_input = batch_original.copy()

        missing_cols, extra_cols = validate_input_df(batch_input)
        if missing_cols:
            st.error(f"Kolom wajib yang belum ada: {missing_cols}")
        else:
            ignored_extra = [c for c in extra_cols if c != label_col]
            if ignored_extra:
                st.info(f"Kolom tambahan akan dibawa di output, tetapi tidak dipakai model: {ignored_extra}")

            batch_numeric = batch_input[RAW_COLUMNS].copy()
            batch_numeric, invalid_rows = sanitize_input_df(batch_numeric)
            if invalid_rows.any():
                st.warning(
                    f"Ada {int(invalid_rows.sum())} baris dengan nilai non-numerik atau kosong pada kolom wajib. "
                    "Nilai itu akan menjadi missing lalu diimputasi jika memungkinkan."
                )

            passthrough = batch_original.copy()
            passthrough[RAW_COLUMNS] = batch_numeric[RAW_COLUMNS]
            result_df, processed_df, selected_df, used_threshold = predict(batch_numeric, bundle, passthrough_df=passthrough)

            summary1, summary2, summary3, summary4 = st.columns(4)
            with summary1:
                metric_card("Jumlah baris", f"{len(result_df):,}", "Data yang berhasil diproses")
            with summary2:
                metric_card("Prediksi Diabetes", f"{int((result_df['prediction_binary'] == 1).sum()):,}", "Jumlah label positif")
            with summary3:
                metric_card("Rata-rata score", f"{float(result_df['model_score_0_100'].mean()):.2f}", "Mean model score 0–100")
            with summary4:
                metric_card("Risiko tinggi", f"{int((result_df['risk_bucket'] == 'Risiko tinggi').sum()):,}", "Count high-risk")

            st.success(f"Prediksi selesai untuk {len(result_df)} baris data.")
            st.caption(f"Threshold keputusan yang digunakan: {used_threshold:.2f}")

            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.markdown("#### Distribusi risk bucket batch")
                st.bar_chart(risk_distribution(result_df["risk_bucket"]).rename("batch_pct"))
            with chart_col2:
                st.markdown("#### Distribusi model score batch")
                st.bar_chart(distribution_from_scores(result_df["model_score_0_100"]).rename("batch_pct"))

            st.markdown("### Perbandingan train vs batch")
            if train_reference and train_reference.get("scores_0_100") is not None:
                score_compare = pd.DataFrame(
                    {
                        "train_pct": distribution_from_scores(train_reference["scores_0_100"]),
                        "batch_pct": distribution_from_scores(result_df["model_score_0_100"]),
                    }
                )
                risk_train_series = train_reference.get("risk_bucket")
                if risk_train_series is None:
                    risk_train_series = [risk_text(s / 100.0, used_threshold) for s in train_reference["scores_0_100"]]
                risk_compare = pd.DataFrame(
                    {
                        "train_pct": risk_distribution(risk_train_series),
                        "batch_pct": risk_distribution(result_df["risk_bucket"]),
                    }
                )

                cmp_col1, cmp_col2 = st.columns(2)
                with cmp_col1:
                    st.markdown("#### Distribusi score: train vs batch")
                    st.bar_chart(score_compare)
                with cmp_col2:
                    st.markdown("#### Distribusi risk bucket: train vs batch")
                    st.bar_chart(risk_compare)

                ref_col1, ref_col2, ref_col3 = st.columns(3)
                with ref_col1:
                    metric_card(
                        "Mean score train",
                        f"{np.mean(train_reference['scores_0_100']):.2f}",
                        "Rata-rata model score pada train-fit reference",
                    )
                with ref_col2:
                    metric_card(
                        "Mean score batch",
                        f"{result_df['model_score_0_100'].mean():.2f}",
                        "Rata-rata model score pada batch saat ini",
                    )
                with ref_col3:
                    gap = float(result_df['model_score_0_100'].mean() - np.mean(train_reference['scores_0_100']))
                    metric_card("Score gap", f"{gap:+.2f}", "Batch mean - train mean")
            else:
                st.info(
                    "Bundle ini belum punya train_reference, jadi grafik train vs batch belum bisa ditampilkan. "
                    "Jalankan ulang export bundle versi train reference."
                )

            if label_col is not None:
                batch_metrics, valid_label_n = compute_metrics_from_probs(result_df[label_col], result_df["probability_diabetes"], used_threshold)
                if batch_metrics is None:
                    st.warning(f"Kolom label '{label_col}' ada, tetapi tidak berisi nilai biner 0/1 yang valid.")
                else:
                    st.markdown("### Perbandingan metrik")
                    metric_rows = []
                    train_metrics = train_reference.get("metrics") if train_reference else None
                    for metric_name in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
                        metric_rows.append(
                            {
                                "metric": metric_name,
                                "train": np.nan if not train_metrics else train_metrics.get(metric_name, np.nan),
                                "batch": batch_metrics.get(metric_name, np.nan),
                            }
                        )
                    metric_df = pd.DataFrame(metric_rows).set_index("metric")
                    st.bar_chart(metric_df)
                    d1, d2, d3 = st.columns(3)
                    with d1:
                        metric_card("Batch label column", label_col, f"Label valid: {valid_label_n} baris")
                    with d2:
                        metric_card("Batch F1", f"{batch_metrics['f1']:.3f}", "Pada batch berlabel")
                    with d3:
                        auc_text = "nan" if np.isnan(batch_metrics["roc_auc"]) else f"{batch_metrics['roc_auc']:.3f}"
                        metric_card("Batch ROC-AUC", auc_text, "Jika kedua kelas tersedia")
            else:
                st.info(
                    "Batch file belum punya kolom label. Jadi yang bisa ditampilkan baru distribusi skor dan distribusi risk bucket, "
                    "belum metrik akurasi/F1 train vs batch."
                )

            display_cols = list(dict.fromkeys([*batch_original.columns.tolist(), "probability_diabetes", "model_score_0_100", "confidence_score_0_100", "risk_bucket", "prediction_label"]))
            st.markdown("#### Hasil prediksi")
            st.dataframe(result_df[display_cols], use_container_width=True, hide_index=True)

            out_csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download hasil prediksi",
                data=out_csv,
                file_name="hasil_prediksi_diabetes.csv",
                mime="text/csv",
                use_container_width=True,
            )

            with st.expander("Lihat preview data setelah preprocessing"):
                st.dataframe(processed_df.head(25), use_container_width=True, hide_index=True)
            with st.expander("Lihat preview fitur akhir yang masuk ke model"):
                st.dataframe(selected_df.head(25), use_container_width=True, hide_index=True)

st.markdown("---")
st.caption(
    "Versi Pro+ menambahkan train-vs-batch comparison. Cocok untuk demo skor individual, batch scoring, dan sanity check generalisasi model."
)
