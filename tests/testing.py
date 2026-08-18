from src.pipelines.training_pipeline import (
    run_training_pipeline,
    prepare_training_data
    )
import pandas as pd
import matplotlib.pyplot as plt
from src.explainability import generate_explanation
from src.visualization import create_shap_summary

result = run_training_pipeline(force=True)
