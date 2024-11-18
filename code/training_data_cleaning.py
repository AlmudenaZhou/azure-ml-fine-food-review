import pandas as pd


def training_dataset_cleaning(data):
    """
    Considerations:

    - Duplicated rows considered by the same userid and text are dropped
    - Id as index
    - ProfileName and Summary columns dropped
    - Since the helpfulness columns are going to be discarded for the model and since they are an insificant number, we will keep the inconsistent rows.
    """

    duplicated_mask = data.sort_values('Time').duplicated(subset=['UserId', 'Text'], keep='last')
    new_dataset = data.drop(duplicated_mask[duplicated_mask].index, axis=0)
    new_dataset.index = new_dataset['Id']
    new_dataset.drop(['ProfileName', 'Summary', 'Id'], axis=1, inplace=True)
    new_dataset['Time'] = pd.to_datetime(data['Time'], unit='s')
    return new_dataset
