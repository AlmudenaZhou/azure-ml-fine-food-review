import os
import logging.config
from dotenv import load_dotenv

from azure.ai.ml import dsl, Input


if __name__ == "__main__":
    import sys

    sys.path.append(os.getcwd())

from src.tools.azure_ml_interface import AzureMLInterface


logging.config.fileConfig('logger.conf')
logger = logging.getLogger(__name__)


@dsl.pipeline(
    compute="serverless",
    description="E2E data_prep-train pipeline",
)
def fine_food_reviews_pipeline(
    pipeline_job_data_input,
    training_data_cleaning_filename="cleaned_data.csv",
    text_processing_filename="processed_data.csv",
    train_split_data_filename="train_data.csv",
    test_split_data_filename="test_data.csv",
    text2vector_filename="text2vec_data.csv",
    text2vector_modelname="text2vector_model.pkl",
    handle_imbalance_filename="handled_imb_data.csv",
    handle_imbalance_modelname="text2vector_model.pkl",
    train_predict_model_filename="predictions.csv",
    train_predict_model_modelname="predictor.pkl",
):

    azure_ml_interface = AzureMLInterface()

    training_data_cleaning_comp = azure_ml_interface.get_component("training_data_cleaning")

    text_processing_comp = azure_ml_interface.get_component("text_processing")

    split_data_comp = azure_ml_interface.get_component("split_data")

    text2vector_comp = azure_ml_interface.get_component("text2vector")

    handle_imbalance_comp = azure_ml_interface.get_component("handle_imbalance")

    train_predict_model_comp = azure_ml_interface.get_component("train_predict_model")

    pipeline_job_data_folder, pipeline_job_data_filename = os.path.split(pipeline_job_data_input.path)

    print("---------------------------------------")
    print(pipeline_job_data_folder, pipeline_job_data_filename)

    training_data_cleaning_job = training_data_cleaning_comp(
        input_data_folder=pipeline_job_data_folder,
        input_data_filename=pipeline_job_data_filename,
        output_data_filename=training_data_cleaning_filename
    )

    text_processing_job = text_processing_comp(
        input_data_folder=training_data_cleaning_job.outputs.output_data_folder,
        input_data_filename=training_data_cleaning_filename,
        output_data_filename=text_processing_filename
    )

    split_data_job = split_data_comp(
        input_data_folder=text_processing_job.outputs.output_data_folder,
        input_data_filename=text_processing_filename,
        train_output_filename=train_split_data_filename,
        test_output_filename=test_split_data_filename
    )

    text2vector_job = text2vector_comp(
        input_data_folder=split_data_job.outputs.output_data_folder,
        input_data_filename=train_split_data_filename,
        output_data_filename=text2vector_filename,
        model_filename=text2vector_modelname,
        is_training="True"
    )

    handle_imbalance_job = handle_imbalance_comp(
        input_data_folder=text2vector_job.outputs.output_data_folder,
        input_data_filename=text2vector_filename,
        output_data_filename=handle_imbalance_filename,
        model_filename=handle_imbalance_modelname,
        is_training="True"
    )

    train_predict_model_job = train_predict_model_comp(
        input_data_folder=handle_imbalance_job.outputs.output_data_folder,
        input_data_filename=handle_imbalance_filename,
        output_data_filename=train_predict_model_filename,
        model_filename=train_predict_model_modelname,
        is_training="True"
    )

    return {
        "predictor_model_folder": train_predict_model_job.outputs.model_output_folder,
        "predictor_data_folder": train_predict_model_job.outputs.output_data_folder
    }


def main():
    subscription_id = os.getenv("SUBSCRIPTION_ID")
    resource_group = os.getenv("RESOURCE_GROUP")
    workspace = os.getenv("WORKSPACE")
    relative_raw_data_uri = os.getenv("RELATIVE_URI_RAW_DATA")

    pipeline_job_data_input = (f"azureml://subscriptions/{subscription_id}/resourcegroups/{resource_group}/workspaces/"
                               + f"{workspace}/datastores/workspaceblobstore/paths/{relative_raw_data_uri}")
    pipeline_job_data_input = Input(type="uri_file", path=pipeline_job_data_input)
    
    pipeline_job = fine_food_reviews_pipeline(
        pipeline_job_data_input
    )

    environment_variables = {"TEST_SIZE": os.getenv("TEST_SIZE"),
                             "SCORING": os.getenv("SCORING"),
                             "TEXT_COLNAME": os.getenv("TEXT_COLNAME"),
                             "TARGET": os.getenv("TARGET")}

    # submit the pipeline job
    azure_ml_interface = AzureMLInterface()
    azure_ml_interface.register_pipeline_from_job_pipeline(pipeline_job, environment_variables)


if __name__ == "__main__":
    load_dotenv()
    main()
