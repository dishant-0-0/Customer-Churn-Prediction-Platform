from src.pipelines.training_pipeline import (
    run_training_pipeline,
    prepare_training_data
    )
import pandas as pd
import matplotlib.pyplot as plt
from src.explainability import generate_explanation
from src.visualization import create_shap_summary

processed = prepare_training_data()

result = run_training_pipeline(force=True)

X_shap = pd.DataFrame(
    processed.X_test_processed,
    columns=processed.feature_names
)
explanation = generate_explanation(
    model=result.model,
    X = X_shap[:100],
)

fig = create_shap_summary(
    explanation
)

plt.show()