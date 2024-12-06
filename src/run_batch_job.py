import os
import logging.config

from dotenv import load_dotenv
from azure.ai.ml import Input

if __name__ == "__main__":
    import sys
    
    sys.path.append(os.getcwd())

from src.tools.azure_ml_interface import AzureMLInterface


logging.config.fileConfig('logger.conf')
logger = logging.getLogger(__name__)


def user_loop(job, azure_ml_interface):
    end = False

    while not end:

        print("""
        Functions (Enter a number from 1 to 3):
              1. Details from the job one time
              2. Wait until the job ends
              3. Exit the script
        """)
        try:
            user_input = int(input())
        except ValueError:
            print("The value must be an integer.")
            continue

        if user_input == 1:
            print(azure_ml_interface.ml_client.jobs.get(job.name))
        elif user_input == 2:
            print("Waiting for the job to finish...")
            results = azure_ml_interface.ml_client.jobs.stream(name=job.name)
            print(results)
            print("Job finished.")
            end = True
        elif user_input == 3:
            print("Exiting the program.")
            end = True
        else:
            print("Only 1-3 numbers are valid.")


def main():

    pipeline_job_input_folder, pipeline_job_input_file = os.path.split(os.getenv("LOCAL_INPUT_PATH"))
    pipeline_job_input_folder = Input(type="uri_folder", path=pipeline_job_input_folder)
    pipeline_job_input_file = Input(type="string", default=pipeline_job_input_file)

    environment_variables = {"TEXT_COLNAME": os.getenv("TEXT_COLNAME"),
                             "TARGET": os.getenv("TARGET")}
    
    batch_endpoint_name = os.getenv("INFERENCE_BATCH_ENDPOINT_NAME")
    azure_ml_interface = AzureMLInterface()
    job = azure_ml_interface.ml_client.batch_endpoints.invoke(
            endpoint_name=batch_endpoint_name,
            inputs={"pipeline_job_input_folder": pipeline_job_input_folder, 
                    "pipeline_job_input_file": pipeline_job_input_file},
            environment_variables=environment_variables
    )

    print("Pipeline job sended to the endpoint.")

    user_loop(job, azure_ml_interface)


        
if __name__ == "__main__":
    load_dotenv()
    main()
