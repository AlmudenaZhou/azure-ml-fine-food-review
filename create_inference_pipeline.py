import os
from urllib.parse import urlparse

from azure.ai.ml.constants import AssetTypes
from azure.ai.ml import dsl, Input

from src.training_pipeline.register_model.register_model_step import RegisterModelStep
from src.tools.azure_ml_interface import AzureMLInterface


@dsl.pipeline(
    compute="serverless",
    description="E2E data_prep-train pipeline",
)
def fine_food_reviews_inference_pipeline(
    pipeline_job_data_input,
    text_processing_filename="processed_data.csv",
    text2vector_filename="text2vec_data.csv",
    model_prediction_filename="predictions.csv"
):
    azure_ml_interface = AzureMLInterface()

    text_processing_comp = azure_ml_interface.get_component("text_processing_inference")

    text2vector_comp = azure_ml_interface.get_component("text2vector_inference")

    model_prediction_comp = azure_ml_interface.get_component("model_prediction_inference")
    
    pipeline_job_data_folder, pipeline_job_data_filename = os.path.split(pipeline_job_data_input.path)

    text_processing_job = text_processing_comp(
        input_data_folder=pipeline_job_data_folder,
        input_data_filename=pipeline_job_data_filename,
        output_data_filename=text_processing_filename
    )
    text2vector_job = text2vector_comp(
        model=Input(type=AssetTypes.CUSTOM_MODEL, path=text2vec_model.id),
        input_data_folder=text_processing_job.outputs.output_data_folder,
        input_data_filename=text_processing_filename,
        output_data_filename=text2vector_filename
    )
    model_prediction_job = model_prediction_comp(
        model=Input(type=AssetTypes.CUSTOM_MODEL, path=pred_model.id),
        input_data_folder=text2vector_job.outputs.output_data_folder,
        input_data_filename=text2vector_filename,
        output_data_filename=model_prediction_filename
    )

    return {"predictions": model_prediction_job.outputs.output_data_folder}



def main():
    subscription_id = os.getenv("SUBSCRIPTION_ID")
    resource_group = os.getenv("RESOURCE_GROUP")
    workspace = os.getenv("WORKSPACE")
    relative_raw_data_uri = os.getenv("RELATIVE_URI_RAW_DATA")

    pipeline_job_data_input = (f"azureml://subscriptions/{subscription_id}/resourcegroups/{resource_group}/workspaces/"
                               + f"{workspace}/datastores/workspaceblobstore/paths/{relative_raw_data_uri}")
    pipeline_job_data_input = Input(type="uri_file", path=pipeline_job_data_input)
    
    pipeline_job = fine_food_reviews_inference_pipeline(pipeline_job_data_input)

    azure_ml_interface = AzureMLInterface()
    azure_ml_interface.register_pipeline_from_job_pipeline(pipeline_job)


if __name__ == "__main__":

    text2vec_model_path = os.getenv("TEXT2VEC_TRAINED_MODEL_PATH")
    pred_model_path = os.getenv("PREDICTOR_TRAINED_MODEL_PATH")

    model_params = {
        "text2vec": {"model_path": text2vec_model_path, "model_name": "fine-food-reviews-text2vec2", 
                     "model_version": "2", "model_type": AssetTypes.CUSTOM_MODEL},
        "pred": {"model_path": pred_model_path, "model_name": "fine-food-reviews-predictor2",
                 "model_version": "2", "model_type": AssetTypes.CUSTOM_MODEL}
    }
    models = RegisterModelStep().main(model_params_dict=model_params)
    text2vec_model, pred_model = models["text2vec"], models["pred"]
    main()
