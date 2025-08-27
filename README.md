![Python](https://img.shields.io/badge/Python-3.8%2B-blue) 
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange) 
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-yellow) 
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

# Website Traffic Prediction using Synthetic Data

> Completed in 2023 as part of my MSc coursework.  
> Publicly released on GitHub in 2025.

This project predicts **website traffic (UniqueVisits)** using multiple AI/ML models.  
The dataset is **synthetically generated** to simulate real-world website traffic patterns.

---

## 🚀 Features
- Implements multiple regression & deep learning models:
  - **LSTM** (TensorFlow/Keras)
  - **Linear Regression**
  - **K-Nearest Neighbors (KNN)**
  - **Support Vector Regression (SVR)**
  - **Multi-layer Perceptron (MLPRegressor)**
- **Hyperparameter tuning** with GridSearchCV
- Evaluation metrics: **MSE** and **MAE**
- Comparative charts for model performance

---

## 📊 Dataset
- Generated via `src/synthetic_data_gen.py`  
- Hourly data from **2017–2022**
- Columns:
  - `Date` – timestamp
  - `PageLoad` – simulated page load events
  - `UniqueVisits` – target variable
  - `FirstVisits` – number of first-time visitors
  - `ReturnVisits` – repeat visitors
  - `DayOfWeek` – categorical feature (e.g., Monday)

⚠️ Note:  
This dataset is **synthetic** (not real).  
It was created to mimic realistic traffic patterns for experimentation.  

**Inspiration:**  
Similar to datasets like the [Google Analytics Sample Dataset](https://support.google.com/analytics/answer/6367342?hl=en) and the [Kaggle Web Traffic Forecasting](https://www.kaggle.com/competitions/web-traffic-time-series-forecasting).  
However, **no real-world data was used**.

---

## 📂 Repository Structure
```
website-traffic-prediction-using-synthetic-data/
├─ src/
│  ├─ AI_Assessment.py        # main training + evaluation pipeline
│  └─ synthetic_data_gen.py   # script to generate synthetic dataset
├─ data/
│  └─ web_traffic_data.csv    # dataset (or sample)
├─ docs/
│  └─ ProjectReport.pdf       # full MSc project report
├─ requirements.txt           # dependencies
├─ .gitignore
├─ LICENSE
└─ README.md
```

---

## ⚙️ Installation
```bash
# 1. Clone the repo
git clone https://github.com/ShouvikDas01/website-traffic-prediction-using-synthetic-data.git
cd website-traffic-prediction-using-synthetic-data

# 2. Create venv (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage
1. **Generate dataset (optional):**
   ```bash
   python src/synthetic_data_gen.py
   ```
   This will create/update `data/web_traffic_data.csv`.

2. **Train and evaluate models:**
   ```bash
   python src/AI_Assessment.py
   ```

The script:
- Runs **EDA** (visualizations, correlations, distributions).  
- Trains all models.  
- Plots predictions vs. actuals.  
- Compares metrics (MSE & MAE).  
- Runs hyperparameter tuning (GridSearchCV & manual search for LSTM).  

---

## 📈 Results (summary)
- **LSTM** → achieved lowest **MSE** after tuning  
- **KNN** → achieved lowest **MAE** on test data  
- Linear, SVR, and MLP performed reasonably but with higher error  
- Visual comparisons show LSTM tracks seasonal/temporal patterns best  

For full results → see `docs/ProjectReport.pdf`.

---

## 📜 Timeline
- Project completed: **2023 (MSc Coursework)**  
- Open-sourced: **2025**

---

## 📄 License
MIT License © 2023 Shouvik Das  
You are free to use, modify, and distribute this project with proper attribution.  
See the [LICENSE](LICENSE) file for details.

---

## ✍️ Author
**Shouvik Das**  
- MSc in Computer Science, 2024  
- GitHub: [ShouvikDas01](https://github.com/ShouvikDas01)
