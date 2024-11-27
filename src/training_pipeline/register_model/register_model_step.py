import os

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential


class RegisterModelStep:

    def __init__(self):
        pass

    def main(self, model_path):
        subscription_id = os.getenv("SUBSCRIPTION_ID")
        resource_group = os.getenv("RESOURCE_GROUP")
        workspace = os.getenv("WORKSPACE")

        ml_client = MLClient(
            DefaultAzureCredential(), subscription_id, resource_group, workspace
        )

        file_model = Model(
            path=model_path,
            type=AssetTypes.CUSTOM_MODEL,
            name="fine_food_reviews_model",
            description="Model to predict whether the review is positive or negative.",
        )
        ml_client.models.create_or_update(file_model)
