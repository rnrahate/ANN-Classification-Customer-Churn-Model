# 🔮 ChurnLens — Customer Churn Prediction Dashboard

> A production-ready, AI-powered customer churn intelligence dashboard built with TensorFlow, Scikit-learn, and Streamlit. Features a modern dark-themed UI with interactive Plotly visualizations and real-time risk classification.

---

## 📸 Dashboard Preview

```
┌─────────────────────────────────────────────────────────────────────┐
│  🔮 ChurnLens  │  Customer Churn Intelligence Dashboard             │
├──────────────┬──────────────────────────────────────────────────────┤
│              │  [KPI Cards]  Churn % │ Retention % │ Risk │ Conf.  │
│  SIDEBAR     │─────────────────────────────────────────────────────│
│              │  [Gauge Chart] [Bar Chart] [Donut Pie]               │
│  · Geography │─────────────────────────────────────────────────────│
│  · Gender    │  [Feature Table]   [Risk Interpretation Panel]       │
│  · Age       │─────────────────────────────────────────────────────│
│  · Credit    │  ▼ Advanced Details (Expander)                       │
│  · Balance   │    Scaled Feature Vector │ Model Architecture Info   │
│  · Salary    │                                                       │
│  · Tenure    │                                                       │
│  · Products  │                                                       │
│  · Cards     │                                                       │
│              │                                                       │
│  [⚡ Predict] │                                                       │
└──────────────┴──────────────────────────────────────────────────────┘
```

---

## ✨ Features

- **Real-time churn probability** prediction using a trained TensorFlow ANN
- **Interactive Plotly charts** — Gauge, Bar, and Donut visualizations
- **3-tier risk classification** — Low / Moderate / High with color-coded UI
- **Dynamic risk interpretation** with actionable business recommendations
- **Horizontal probability progress bar** with conditional coloring
- **KPI metric cards** — Churn %, Retention %, Risk Level, Model Confidence
- **Expandable advanced panel** — raw & scaled feature vectors, model metadata
- **Sidebar input form** — fully grouped and labeled for UX clarity
- **Custom CSS theming** — dark dashboard aesthetic with Space Mono + DM Sans fonts
- **Session state persistence** — results preserved across widget interactions
- **Cached model loading** — `@st.cache_resource` for fast repeat loads

---

## 🗂 Project Structure

```
churnlens/
│
├── churn_dashboard.py                  # Main Streamlit application
│
├── trained_model/
│   └── churn_model.h5                  # Trained TensorFlow/Keras model
│
├── preprocessing_models/
│   ├── one_hot_encoder_geography.pkl   # OneHotEncoder for Geography
│   ├── label_encoder_gender.pkl        # LabelEncoder for Gender
│   └── scaler_features.pkl            # StandardScaler for all features
│
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/churnlens.git
cd churnlens
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Your Model & Preprocessors

Ensure the following files exist in their respective directories before running the app:

| File | Path |
|------|------|
| Keras model | `trained_model/churn_model.h5` |
| OHE encoder | `preprocessing_models/one_hot_encoder_geography.pkl` |
| Label encoder | `preprocessing_models/label_encoder_gender.pkl` |
| Feature scaler | `preprocessing_models/scaler_features.pkl` |

### 5. Run the App

```bash
streamlit run churn_dashboard.py
```

The app will open at **http://localhost:8501**

---

## 📦 Requirements

```txt
streamlit>=1.32.0
tensorflow>=2.12.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
```

Generate `requirements.txt` automatically with:

```bash
pip freeze > requirements.txt
```

---

## 🧠 Model & Preprocessing Details

### Input Features

| Feature | Type | Encoding |
|---------|------|----------|
| `CreditScore` | Numeric | StandardScaler |
| `Geography` | Categorical | OneHotEncoder → France / Germany / Spain |
| `Gender` | Categorical | LabelEncoder → 0 / 1 |
| `Age` | Numeric | StandardScaler |
| `Tenure` | Numeric | StandardScaler |
| `Balance` | Numeric | StandardScaler |
| `NumOfProducts` | Numeric | StandardScaler |
| `HasCrCard` | Binary | StandardScaler |
| `IsActiveMember` | Binary | StandardScaler |
| `EstimatedSalary` | Numeric | StandardScaler |

### Feature Order (must match training pipeline)

```python
FEATURE_ORDER = [
    'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance',
    'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
    'Geography_France', 'Geography_Germany', 'Geography_Spain'
]
```

> ⚠️ **Important:** This column order must exactly match the order used when fitting the `StandardScaler` during training. Any mismatch will silently produce incorrect predictions.

### Model Architecture

| Property | Value |
|----------|-------|
| Framework | TensorFlow / Keras |
| File format | `.h5` |
| Output activation | Sigmoid |
| Task | Binary classification (churn / no churn) |
| Output range | 0.0 → 1.0 (probability) |

---

## 🎯 Risk Classification

Predictions are bucketed into three risk tiers based on churn probability:

| Probability Range | Risk Tier | UI Color | Recommended Action |
|-------------------|-----------|----------|--------------------|
| 0% – 40% | 🟢 Low Risk | Green `#3fb950` | Cross-sell / upsell opportunities |
| 40% – 65% | 🟡 Moderate Risk | Amber `#d29922` | Light-touch engagement campaigns |
| 65% – 100% | 🔴 High Risk | Red `#f85149` | Immediate retention intervention |

> The decision boundary is set at **0.50** by default (standard sigmoid threshold).  
> Model confidence is calculated as the normalized distance from this boundary:  
> `confidence = |prob − 0.5| / 0.5 × 100%`

---

## 🖥 UI Architecture

| Section | Component | Description |
|---------|-----------|-------------|
| Hero Header | Custom HTML/CSS | App title, subtitle, animated gradient card |
| Sidebar | `st.sidebar` | 10 customer input fields, grouped by category |
| KPI Row | `st.metric` | Churn %, Retention %, Risk Level, Confidence |
| Progress Bar | Custom CSS | Gradient fill driven by probability value |
| Risk Badge | Custom HTML | Color-coded inline badge with tier label |
| Gauge Chart | `plotly.graph_objects` | Indicator with 3-zone arc and threshold line |
| Bar Chart | `plotly.graph_objects` | Churn vs Retention vertical bars |
| Donut Chart | `plotly.graph_objects` | Probability distribution with center annotation |
| Feature Table | Custom HTML | Raw input key-value pairs |
| Interpretation | Custom HTML | Tier-specific business recommendation card |
| Threshold Info | Custom HTML | Pill badges explaining all 3 thresholds |
| Advanced Panel | `st.expander` | Scaled vector, raw features, model metadata |

---

## 🚀 Deployment

### Deploy to Streamlit Community Cloud

1. Push your repository to GitHub (ensure model files are included or loaded from cloud storage)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path** to `churn_dashboard.py`
5. Click **Deploy**

> ⚠️ If your `.h5` model is large (>100MB), consider storing it in cloud storage (S3, GCS) and loading it at runtime, or use Git LFS.

### Deploy with Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "churn_dashboard.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t churnlens .
docker run -p 8501:8501 churnlens
```

---

## 🔧 Customization

### Change the Risk Thresholds

Edit these values in `churn_dashboard.py`:

```python
if prob >= 0.65:        # ← High risk threshold
    risk_tier = "HIGH"
elif prob >= 0.40:      # ← Medium risk threshold
    risk_tier = "MEDIUM"
else:
    risk_tier = "LOW"
```

### Swap the Color Palette

All risk colors are defined as local variables and cascade to every chart and badge:

```python
risk_color = "#f85149"   # High  — change to any hex
risk_color = "#d29922"   # Med   — change to any hex
risk_color = "#3fb950"   # Low   — change to any hex
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Streamlit](https://streamlit.io/) — rapid ML app framework
- [TensorFlow / Keras](https://www.tensorflow.org/) — deep learning backend
- [Plotly](https://plotly.com/python/) — interactive chart library
- [Scikit-learn](https://scikit-learn.org/) — preprocessing utilities
- Dataset inspired by the [Churn Modelling Dataset](https://www.kaggle.com/datasets/shubh0799/churn-modelling) on Kaggle

---

<div align="center">
  <strong>ChurnLens</strong> · Built with TensorFlow + Streamlit<br>
  <sub>For internal analytical use · Not intended as sole basis for business decisions</sub>
</div>