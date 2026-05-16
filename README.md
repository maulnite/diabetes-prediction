# DiabetesAI Clinical Dashboard

DiabetesAI Clinical Dashboard adalah aplikasi web untuk memprediksi risiko diabetes menggunakan model Machine Learning berbasis Flask. Aplikasi ini menyediakan prediksi manual, prediksi batch melalui CSV, visualisasi performa model, riwayat prediksi, dan endpoint API sederhana.

## Fitur Utama

- Prediksi risiko diabetes secara manual
- Upload CSV untuk batch prediction
- Hasil prediksi dengan probabilitas dan kategori risiko
- Riwayat prediksi
- Visualisasi model performance
- Feature importance dan confusion matrix
- Download hasil prediksi
- Dark mode / light mode
- Deploy menggunakan Docker dan Docker Compose

## Teknologi yang Digunakan

- Python
- Flask
- Bootstrap
- Scikit-learn
- Pandas
- NumPy
- Chart.js
- Docker

## Struktur Project

```text
diabetes-flask-bootstrap/
├── app.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── model/
│   └── diabetes_model_bundle_revised.pkl
├── data/
│   └── diabetes.csv
├── static/
│   ├── css/
│   └── js/
├── templates/
└── README.md
```

## Cara Menjalankan dengan Docker Compose

Pastikan Docker Desktop sudah aktif, lalu jalankan perintah berikut dari folder project:

```bash
docker compose up -d --build
```

Setelah proses selesai, buka aplikasi di browser:

```text
http://localhost:5000
```

Untuk menghentikan aplikasi:

```bash
docker compose down
```

Untuk melihat log aplikasi:

```bash
docker compose logs -f
```

## Cara Menjalankan Tanpa Docker

Install dependency terlebih dahulu:

```bash
pip install -r requirements.txt
```

Jalankan aplikasi:

```bash
python app.py
```

Lalu buka:

```text
http://localhost:5000
```

## Format CSV untuk Batch Prediction

File CSV harus memiliki kolom berikut:

```csv
Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
```

Kolom `Outcome` tidak wajib ada karena hanya digunakan pada dataset training/evaluasi.

## Dataset

Model pada aplikasi ini dilatih menggunakan **Pima Indians Diabetes Database**. Dataset ini berisi data medis pasien seperti jumlah kehamilan, kadar glukosa, tekanan darah, insulin, BMI, diabetes pedigree function, usia, dan label diabetes.

## Catatan

Aplikasi ini dibuat untuk kebutuhan pembelajaran dan demonstrasi Machine Learning. Hasil prediksi bukan diagnosis medis resmi. Untuk keputusan medis, tetap konsultasikan dengan tenaga kesehatan profesional.
