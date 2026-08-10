from src.pipelines.training_pipeline import run_training_pipeline
import matplotlib.pyplot as plt
from src.visualization import create_confusion_matrix

result = run_training_pipeline()
fig = create_confusion_matrix(
    result.evaluation
)

plt.show()