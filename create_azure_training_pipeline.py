import os
import logging.config
from dotenv import load_dotenv

from azure.ai.ml import dsl, Input

from src.tools.azure_ml_interface import AzureMLInterface


logging.config.fileConfig('logger.conf')
logger = logging.getLogger(__name__)


@dsl.pipeline(
    compute="serverless",
    description="E2E data_prep-train pipeline",
)
def fine_food_reviews_pipeline(
    pipeline_job_data_input,
    pipeline_job_registered_model_name,
    cleaned_data_filename="cleaned_data.csv",
    processed_data_filename="processedprocessed_data.csv",
    x_train_filename="X_train.csv",
    x_test_filename="X_test.csv",
    y_train_filename="y_train.csv",
    y_test_filename="y_test.csv",
    text2vector_data_filename="text2vect_X_train.csv",
    imb_x_data_filename="imb_X_train.csv",
    imb_y_data_filename="imb_y_train.csv",
):

    azure_ml_interface = AzureMLInterface()

    training_data_cleaning_comp = azure_ml_interface.get_component("training_data_cleaning")

    text_processing_comp = azure_ml_interface.get_component("text_processing")

    split_data_comp = azure_ml_interface.get_component("split_data")

    text2vector_comp = azure_ml_interface.get_component("text2vector")

    handle_imbalance_comp = azure_ml_interface.get_component("handle_imbalance")

    model_training_comp = azure_ml_interface.get_component("model_training")

    # using data_prep_function like a python call with its own inputs
    training_data_cleaning_job = training_data_cleaning_comp(
        data=pipeline_job_data_input,
        clean_filename=cleaned_data_filename
    )
    
    text_processing_job = text_processing_comp(
        input_data_folder=training_data_cleaning_job.outputs.clean_data,
        input_data_filename=cleaned_data_filename,
        processed_filename=processed_data_filename
    )

    # using train_func like a python call with its own inputs
    split_data_job = split_data_comp(
        input_data_folder=text_processing_job.outputs.processed_data,
        input_data_filename=processed_data_filename,
        x_train_filename=x_train_filename,
        x_test_filename=x_test_filename,
        y_train_filename=y_train_filename,
        y_test_filename=y_test_filename,
    )

    text2vector_job = text2vector_comp(
        input_data_folder=split_data_job.outputs.split_data,
        input_x_filename=x_train_filename,
        input_y_filename=y_train_filename,
        text2vec_data_filename=text2vector_data_filename
    )

    handle_imbalance_job = handle_imbalance_comp(
        input_data_folder=text2vector_job.outputs.output_folder_path,
        input_x_filename=text2vector_data_filename,
        input_y_filename=y_train_filename,
        imb_x_data_filename=imb_x_data_filename,
        imb_y_data_filename=imb_y_data_filename,
    )

    model_training_job = model_training_comp(
        input_data_folder=handle_imbalance_job.outputs.output_folder_path,
        input_x_filename=imb_x_data_filename,
        input_y_filename=imb_y_data_filename,
    )

    # a pipeline returns a dictionary of outputs
    # keys will code for the pipeline output identifier
    return {
        "predictor_model_path": model_training_job.outputs.model_path
    }


def main():
    subscription_id = os.getenv("SUBSCRIPTION_ID")
    resource_group = os.getenv("RESOURCE_GROUP")
    workspace = os.getenv("WORKSPACE")
    relative_raw_data_uri = os.getenv("RELATIVE_URI_RAW_DATA")

    pipeline_job_data_input = (f"azureml://subscriptions/{subscription_id}/resourcegroups/{resource_group}/workspaces/"
                               + f"{workspace}/datastores/workspaceblobstore/paths/{relative_raw_data_uri}")
    pipeline_job_data_input = Input(type="uri_file", path=pipeline_job_data_input)

    pipeline_job_registered_model_name = "fine_food_review_model"
    
    pipeline_job = fine_food_reviews_pipeline(
        pipeline_job_data_input,
        pipeline_job_registered_model_name
    )

    environment_variables = {"SUBSCRIPTION_ID": subscription_id,
                             "RESOURCE_GROUP": resource_group,
                             "WORKSPACE": workspace,
                             "TEST_SIZE": os.getenv("TEST_SIZE"),
                             "SCORING": os.getenv("SCORING"),}

    # submit the pipeline job
    azure_ml_interface = AzureMLInterface()
    azure_ml_interface.register_pipeline_from_job_pipeline(pipeline_job, environment_variables)


if __name__ == "__main__":
    load_dotenv()
    main()
