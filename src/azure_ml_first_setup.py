import os
import sys

from dotenv import load_dotenv

if __name__ == "__main__":
    sys.path.append(os.getcwd())

from src.azure_ml_interface import AzureMLInterface


load_dotenv()
        

def main():
    subscription_id = os.getenv("SUBSCRIPTION_ID")
    resource_group = os.getenv("RESOURCE_GROUP")
    workspace = os.getenv("WORKSPACE")
    uri_filename = os.getenv("URI_FILENAME")
    ci_basic_name = os.getenv("COMPUTE_INSTANCE_NAME")

    local_filepath = "data/reviews_short.csv"
    print("Conecting MLClient")
    azure_ml_interface = AzureMLInterface(subscription_id, resource_group, workspace)
    print("Creating URI file...")
    azure_ml_interface.create_urifile_dataasset_from_local_file(local_filepath,
                                                                description="Fine Food Reviews from Amazon", 
                                                                name=uri_filename,
                                                                version="0")
    print("Creating Compute instance...")
    azure_ml_interface.create_compute_instance(ci_basic_name, ci_size="Standard_DS11_v2")
    print("Creating Environment")
    environment_name = os.getenv("AZURE_ML_ENVIRONMENT_NAME")
    azure_ml_interface.create_environment_from_dockerfile(environment_name=environment_name)


if __name__ == "__main__":
    main()
