from src.models.registry import get_model
from src.models.train import train_model
from src.models.evaluate import evaluate_model
from src.models.predict import predict
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

predictions = predict(
    model = model,
    preprocessor= processed_data.preprocessor,
    X = processed_data.X_test
)

print(results.accuracy)
print(results.recall)
print(predictions[:5])