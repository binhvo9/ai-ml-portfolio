# 🌸 Iris Flower Classification — SVC + GridSearchCV
A recruitment-ready ML project using a clean **Pipeline (StandardScaler → SVC)** and **GridSearchCV** with 5-fold CV.

## 🧠 Highlights
- End-to-end: load data → pipeline → GridSearchCV → evaluate → save artifacts & metrics
- Reproducible structure, CLI prediction, optional Streamlit app
- Clear metrics + confusion matrix for quick validation

📊 Results (example)

Best CV Acc: ~0.97–1.00 (5-fold)

Test Acc: see results/metrics.json

Model: artifacts/iris_svc_pipeline.joblib

🧪 Tech Stack

scikit-learn · Pipeline · GridSearchCV · StratifiedKFold · joblib

Project Structure
iris-classification-gridsearch/
├─ artifacts/                # saved model (joblib)
├─ results/                  # metrics.json
├─ src/
│  ├─ train.py               # train + grid search + save
│  └─ predict.py             # CLI prediction
├─ app.py                    # (optional) Streamlit UI
├─ requirements.txt
├─ .gitignore
└─ README.md


## 🚀 Quickstart
```bash
pip install -r requirements.txt
cd src
python train.py                # trains & saves artifacts/ + results/
python predict.py 5.1 3.5 1.4 0.2


