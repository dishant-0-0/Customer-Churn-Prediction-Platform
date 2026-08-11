from src.pipelines.training_pipeline import run_training_pipeline
import matplotlib.pyplot as plt
from src.visualization import create_roc_curve

result = run_training_pipeline()
fig = create_roc_curve(
    result.evaluation
)

plt.show()