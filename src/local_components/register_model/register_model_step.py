import os

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential


class RegisterModelStep:

    def __init__(self):
        subscription_id = os.getenv("SUBSCRIPTION_ID")
        resource_group = os.getenv("RESOURCE_GROUP")
        workspace = os.getenv("WORKSPACE")
        self.ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace)

    def register_model(self, model_path, model_name, model_version, model_type):

        file_model = Model(
            path=model_path,
            type=model_type,
            name=model_name,
            description="Model to predict whether the review is positive or negative.",
            version=model_version
        )
        model = self.ml_client.models.create_or_update(file_model)
        return model

    def main(self, model_params_dict: dict):
        models = {}
        for model_name, model_params in model_params_dict.items():
            model = self.register_model(**model_params)
            models[model_name] = model
        return models
