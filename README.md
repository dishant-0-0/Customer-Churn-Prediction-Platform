# Customer Churn Prediction Platform

A classical machine learning phase-1 project for predicting customer churn, with data cleaning, model training, comparison, explainability, and deployment.

## Phase 1: Classical ML Foundation (2–4 weeks)

### Skills
- Data cleaning
- Feature engineering
- Model training
- Evaluation
- Explainability

### Roadmap

#### Week 1
- Collect telecom/banking churn dataset
- Perform exploratory data analysis (EDA)
- Handle missing values
- Create features and encode categorical variables

#### Week 2
- Train models:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Compare metrics and select the best model

#### Week 3
- Add SHAP explainability
- Build FastAPI endpoint for inference

#### Week 4
- Dockerize the API
- Deploy the service

## Project structure

- `data/` – raw and processed datasets
- `notebooks/` – EDA and experimentation notebooks
- `src/` – reusable Python modules
- `Dockerfile` – container image definition
- `requirements.txt` – Python dependencies

## Getting started

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the API locally:
   ```bash
   uvicorn src.api:app --host 0.0.0.0 --port 8000
   ```

## Notes

- Add your dataset files to `data/`
- Use `notebooks/` for EDA and model comparison
- Extend `src/` with feature engineering, model persistence, and deployment logic
