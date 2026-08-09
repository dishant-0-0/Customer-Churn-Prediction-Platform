from src.pipelines.training_pipeline import prepare_training_data
from src.pipelines.inference_pipeline import run_inference

processed = prepare_training_data()

predictions_noproba = run_inference(
    processed.X_test.head(10)
)

predictions_proba = run_inference(
    processed.X_test.head(10),
    return_probabilities= True
)

print(predictions_noproba)
print(predictions_proba)