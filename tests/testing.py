from src.models.registry import get_model
from src.models.train import train_model
from src.models.evaluate import evaluate_model
from src.pipelines.training_pipeline import prepare_training_data

processed_data = prepare_training_data()

model = get_model()

trained_model = train_model(
    processed= processed_data,
    model= model,
)

results = evaluate_model(
    processed= processed_data,
    model= trained_model
)

print(results.accuracy)
print(results.precision)
print(results.recall)
print(results.f1)
print(results.roc_auc)
print(results.confusion_matrix)