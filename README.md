# Titanic Survival Prediction 🚢

A machine learning pipeline to predict passenger survival on the Titanic using the [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic) dataset.

## Results
| Metric | Score |
|--------|-------|
| Validation Accuracy | ~86.6% |
| Validation MAE | 0.134 |
| Model | XGBoost Classifier |

## Approach
Built an end-to-end sklearn Pipeline covering the full ML workflow:
- Automatically split features into numerical and categorical columns
- Imputed missing numerical values with constant strategy
- Imputed missing categorical values with most-frequent, then applied One-Hot Encoding
- Trained XGBoost classifier on 80% training data, validated on 20%
- Generated predictions on test set and exported to CSV for Kaggle submission

## Tech Stack
![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-red)
![Pandas](https://img.shields.io/badge/Pandas-green)

## Project Structure
titanic-survival-prediction/
│
├── Titanic-ML.ipynb     # Main notebook with full pipeline
├── submission.csv       # Kaggle submission file
└── README.md

## What I Learned
- Building a full preprocessing + model Pipeline with `ColumnTransformer`
- Handling mixed data types (numerical + categorical) cleanly
- XGBoost for tabular classification problems
- End-to-end ML workflow: load → clean → encode → train → evaluate → predict

## How to Run
1. Clone the repo
```bash
   git clone https://github.com/YOUR_USERNAME/titanic-survival-prediction
```
2. Install dependencies
```bash
   pip install pandas scikit-learn xgboost
```
3. Open `titanic.ipynb` in Jupyter or Kaggle Notebooks and run all cells
