import os
import logging
import time

from src.tools.azure_ml_interface import AzureMLInterface


logger = logging.getLogger(__name__)

def manage_compute_instance_starting():

    azure_ml_interface = AzureMLInterface()

    compute_name = os.getenv("COMPUTE_INSTANCE_NAME")
    comp_status = azure_ml_interface.get_compute_status(compute_name)
    logger.info("Compute instance Status:", comp_status.state)

    if comp_status.state == 'Updating':
        while comp_status.state not in ['Stopped', 'Running']:
            comp_status = azure_ml_interface.get_compute_status(compute_name)
            time.sleep(2)

    if comp_status.state == 'Stopped':
        azure_ml_interface.start_compute(compute_name)
        while comp_status.state != "Running":
            comp_status = azure_ml_interface.get_compute_status(compute_name)
            time.sleep(2)