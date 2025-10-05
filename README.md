<!-- PROJECT HEADER -->
<p align="center">
  <img src="https://img.icons8.com/external-flaticons-lineal-color-flat-icons/512/external-brain-anatomy-flaticons-lineal-color-flat-icons.png" width="120" alt="Brain icon"/>
</p>

<h1 align="center">🧠 Parkinson Disease Predictor</h1>

<p align="center">
  <em>Machine Learning–based voice analysis for early detection of Parkinson’s symptoms.</em><br>
  <strong>Built with Streamlit • scikit-learn • Python</strong>
</p>

---

## 🌟 Overview

**Parkinson Disease Predictor** is an interactive **Streamlit web app** that uses a trained **Random Forest Classifier** to predict the likelihood of Parkinson’s disease from **biomedical voice features**.

By analyzing features such as jitter, shimmer, RPDE, and spread1/spread2, the app provides an easy-to-understand probability score and classification result.

> ⚠️ **Note:** This app is for **educational and research purposes only** — not for medical diagnosis.

---

## 🧩 Features

- 🎛️ **Interactive UI** – User-friendly Streamlit interface  
- 📈 **ML Model (Random Forest)** – Achieved 93% accuracy  
- 📊 **Feature Importance Chart** – Visualizes top contributing features  
- 🧮 **Preprocessing Pipeline** – Scaler applied automatically  
- 📁 **Batch Predictions** – Supports CSV uploads for multiple records  

---

## 📁 Project Structure
```
📦 Parkinson_Disease_Predictor
┣ 📂 artifacts/ # Trained model and related metadata
┃ ┣ parkinsons_model.pkl
┃ ┣ feature_names.json
┃ ┣ feature_stats.json
┃ ┗ scaler.pkl
┣ 📜 streamlit_app.py # Streamlit web application
┣ 📜 Parkinson_disease_prediction.ipynb # Notebook used for training & evaluation
┣ 📜 requirements.txt
┣ 📜 .gitignore
┗ 📜 README.md
```

---

## ⚙️ Installation & Usage

### 1️⃣ Clone the Repository
```
git clone https://github.com/NakulLimbani/Parkinson_Disease_Predictor.git
cd Parkinson_Disease_Predictor
```
2️⃣ Create and Activate a Virtual Environment
```
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```
3️⃣ Install Dependencies
```
pip install -r requirements.txt
```
4️⃣ Run the Streamlit App
```
streamlit run streamlit_app.py
```
Open the link in your browser (default: http://localhost:8501).

---

## 🧠 Model Performance

| Metric     | Score  |
|-------------|--------|
| Accuracy    | 93.88% |
| Precision   | 92.68% |
| Recall      | 99.00%   |
| F1-Score    | 96.20% |

**Model:** RandomForestClassifier  
**Framework:** scikit-learn  
**Language:** Python 3.10+

---

🧪 Dataset Details
- Source: UCI Machine Learning Repository – Parkinson’s Dataset
- Instances: 195 voice recordings
- Features: 22 biomedical voice measurements
- Target Variable: status → 1 = Parkinson’s, 0 = Healthy

---

🧰 Tech Stack
- 🐍 Python
- 🎨 Streamlit
- 🤖 scikit-learn
- 📊 pandas
- 📊 numpy
- 📈 matplotlib

---

👨‍💻 Author
Nakul Limbani
