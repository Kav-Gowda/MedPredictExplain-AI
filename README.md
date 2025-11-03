# MedPredictExplain-AI

Explainable ML on the classic **UCI Heart Disease** dataset, with a focus on **Partial Dependence Plots (PDPs)** to interpret model behavior. You’ll train baseline models (Logistic Regression, Random Forest), evaluate them, and generate PDPs (1D & 2D) to see how features like age, cholesterol, and max heart rate influence predictions.

---

## 🚀 What this project does
- Loads and cleans the UCI Heart Disease dataset (via `ucimlrepo`)
- Handles missing values (mean imputation for `ca` and `thal`)
- Trains **Logistic Regression** and **Random Forest** classifiers
- Evaluates on a stratified train/test split with classification reports + confusion matrices
- Implements **manual PDP** for one feature
- Uses **scikit-learn’s** `PartialDependenceDisplay` for robust PDPs (1D and 2D)

---

## 🧠 Tech Stack
Python · pandas · numpy · scikit-learn · matplotlib · seaborn · ucimlrepo · tqdm

---

## 📦 Setup
```bash
git clone https://github.com/<your-username>/MedPredictExplain-AI.git
cd MedPredictExplain-AI
pip install -r requirements.txt
```

---

## ▶️ Run
1. Open the Colab notebook or run the `.py` script.
2. The project will:
   - Fetch the UCI Heart Disease dataset using `ucimlrepo`
   - Impute missing values (`ca`, `thal`)
   - Train both Logistic Regression and Random Forest models
   - Evaluate and visualize:
     - Classification reports
     - Confusion matrices
     - Model coefficients / feature importances
     - PDPs (1D + 2D)
3. Review outputs and interpret how each feature impacts model predictions.

---

## 🧪 Dataset Info
**UCI Heart Disease Dataset**
- 303 samples × 13 features
- Binary target (0 = No disease, 1 = Disease)
- Features: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal  
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+Disease)

---

## 🧩 Why PDP?
Partial Dependence Plots show the **average marginal effect** of a feature on model predictions, while holding others constant.  
They’re model-agnostic, intuitive, and ideal for **interpreting medical ML models**, where explainability matters more than raw accuracy.

---

## 📈 Example Outputs
- **Classification Report**
- **Confusion Matrices** (for both models)
- **Feature Importance Bar Charts**
- **1D PDP:** How `age` influences predicted heart disease risk
- **2D PDP:** Interaction of `thalach` and `chol`

---

## 📄 License
MIT License - see `LICENSE` for details.
