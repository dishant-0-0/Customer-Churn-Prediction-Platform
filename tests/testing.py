from src.models.registry import get_model
from src.models.train import train_model
from src.models.evaluate import evaluate_model
from src.models.predict import predict
from src.pipelines.training_pipeline import prepare_training_data
from src. persistence import InferenceArtifacts
from src.persistence.save import save_artifacts
from src.persistence.load import load_artifacts

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

# artifacts = InferenceArtifacts(
#     model= model,
#     preprocessor= processed_data.preprocessor,
#     feature_names= processed_data.feature_names
# )

# save_path = save_artifacts(
#     artifacts=artifacts
# )

# print(save_path)

loaded_artifacts = load_artifacts()

print(type(loaded_artifacts.model))
print(type(loaded_artifacts.preprocessor))
print(type(loaded_artifacts.feature_names[:5]))