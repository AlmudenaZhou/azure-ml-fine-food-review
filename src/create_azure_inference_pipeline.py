import os

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
    text_processing_filename="infer_processed_data.csv",
    text2vector_filename="infer_text2vec_data.csv",
    text2vector_modelname="text2vector_model.pkl",
    train_predict_model_filename="infer_predictions.csv",
    train_predict_model_modelname="predictor.pkl",
):
    azure_ml_interface = AzureMLInterface()

    text_processing_comp = azure_ml_interface.get_component("text_processing")

    text2vector_comp = azure_ml_interface.get_component("text2vector")

    train_predict_model_comp = azure_ml_interface.get_component("train_predict_model")
    
    pipeline_job_data_folder, pipeline_job_data_filename = os.path.split(pipeline_job_data_input.path)

    print("---------------------------------------")
    print(pipeline_job_data_folder, pipeline_job_data_filename)

    text_processing_job = text_processing_comp(
        input_data_folder=pipeline_job_data_folder,
        input_data_filename=pipeline_job_data_filename,
        output_data_filename=text_processing_filename
    )

    print()
    text2vector_job = text2vector_comp(
        model_input_path=Input(type=AssetTypes.CUSTOM_MODEL, path=text2vec_model.id),
        model_filename=text2vector_modelname,
        input_data_folder=text_processing_job.outputs.output_data_folder,
        input_data_filename=text_processing_filename,
        output_data_filename=text2vector_filename,
        is_training="False"
    )

    model_prediction_job = train_predict_model_comp(
        model_input_path=Input(type=AssetTypes.CUSTOM_MODEL, path=pred_model.id),
        model_filename=train_predict_model_modelname,
        input_data_folder=text2vector_job.outputs.output_data_folder,
        input_data_filename=text2vector_filename,
        output_data_filename=train_predict_model_filename,
        is_training="False"
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

    environment_variables = {"TEXT_COLNAME": os.getenv("TEXT_COLNAME"),
                             "TARGET": os.getenv("TARGET")}

    azure_ml_interface = AzureMLInterface()
    azure_ml_interface.register_pipeline_from_job_pipeline(pipeline_job, environment_variables=environment_variables)


if __name__ == "__main__":

    text2vec_model_path = os.getenv("TEXT2VEC_TRAINED_MODEL_PATH")
    pred_model_path = os.getenv("PREDICTOR_TRAINED_MODEL_PATH")

    model_params = {
        "text2vec": {"model_path": text2vec_model_path, "model_name": "fine-food-reviews-text2vec", 
                     "model_version": "3", "model_type": AssetTypes.CUSTOM_MODEL},
        "pred": {"model_path": pred_model_path, "model_name": "fine-food-reviews-predictor",
                 "model_version": "3", "model_type": AssetTypes.CUSTOM_MODEL}
    }
    models = RegisterModelStep().main(model_params_dict=model_params)
    text2vec_model, pred_model = models["text2vec"], models["pred"]
    main()
