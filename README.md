# Customer Churn Prediction Platform

A project for predicting customer churn, with data cleaning, model training, comparison, explainability, and deployment.

### Roadmap

- Collect telecom/banking churn dataset
- Perform exploratory data analysis (EDA)
- Handle missing values
- Create features and encode categorical variables

- Train models:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Compare metrics and select the best model

- Add SHAP explainability
- Build FastAPI endpoint for inference

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
