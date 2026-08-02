from src.models.registry import get_model
from src.models.train import train_model
from src.pipelines.training_pipeline import prepare_training_data

processed_data = prepare_training_data()

model = get_model()

trained_model = train_model(
    processed= processed_data,
    model= model,
)
print(type(trained_model))