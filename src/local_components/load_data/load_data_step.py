import os
import logging

import pandas as pd

from tools.azure_ml_interface import AzureMLInterface


logger = logging.getLogger(__name__)

class LoadDataStep:

    def __init__(self, input_data_uri=None):
        if input_data_uri:
            self.azure_ml_interface = AzureMLInterface(subscription_id, resource_group, workspace)
            self.raw_data_uri = input_data_uri
            return 

        relative_raw_data_uri = os.getenv("RELATIVE_URI_RAW_DATA")
        if os.getenv("ENVIRONMENT") == "local":
            filename = relative_raw_data_uri.split('/')[-1]
            self.raw_data_uri = os.path.join("data", filename)
        else:
            subscription_id = os.getenv("SUBSCRIPTION_ID")
            resource_group = os.getenv("RESOURCE_GROUP")
            workspace = os.getenv("WORKSPACE")

            self.raw_data_uri = (f"azureml://subscriptions/{subscription_id}/resourcegroups/{resource_group}/workspaces/" +
                                f"{workspace}/datastores/workspaceblobstore/paths/{relative_raw_data_uri}")

            self.azure_ml_interface = AzureMLInterface(subscription_id, resource_group, workspace)

    def main(self):
        logger.info(f"Reading {self.raw_data_uri}")
        raw_data = pd.read_csv(self.raw_data_uri)
        return raw_data
