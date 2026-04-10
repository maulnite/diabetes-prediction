class DiabetesInferencePipeline:
    REQUIRED_COLS = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]

    def __init__(self, preprocessor, model, threshold, feature_names, model_name):
        self.preprocessor  = preprocessor
        self.model         = model
        self.threshold     = threshold
        self.feature_names = feature_names
        self.model_name    = model_name

    def _validate(self, df):
        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f'Kolom tidak ditemukan: {missing}')
        return df[self.REQUIRED_COLS].copy()

    def predict(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Prediksi dari raw DataFrame — terima data mentah langsung."""
        df_val = self._validate(raw_df)
        X_proc = self.preprocessor.transform(df_val)
        probs  = self.model.predict_proba(X_proc)[:, 1]
        preds  = (probs >= self.threshold).astype(int)
        return pd.DataFrame({
            'probability': probs.round(4),
            'prediction':  preds,
            'label':       ['Diabetes' if p else 'Tidak Diabetes' for p in preds],
            'confidence':  [
                'Tinggi' if abs(prob - 0.5) > 0.25 else
                'Sedang' if abs(prob - 0.5) > 0.10 else 'Rendah'
                for prob in probs
            ]
        })

    def predict_proba(self, raw_df: pd.DataFrame) -> np.ndarray:
        df_val = self._validate(raw_df)
        return self.model.predict_proba(self.preprocessor.transform(df_val))

    def __repr__(self):
        return (f'DiabetesInferencePipeline(model={self.model_name}, '
                f'threshold={self.threshold:.4f}, n_features={len(self.feature_names)})')
