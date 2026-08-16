from src.pipelines.training_pipeline import run_training_pipeline
import matplotlib.pyplot as plt
from src.visualization import create_feature_importance

result = run_training_pipeline(force=True)
fig = create_feature_importance(
    result.model,
    result.artifacts.feature_names
)

plt.show()