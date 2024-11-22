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


def data_to_binary(cleaned_data):
    df_binary = pd.DataFrame(cleaned_data, columns=['Score', 'Text'])

    def label_score(score):
        if int(score) >= 4:
            return 1
        elif int(score) <= 2:
            return 0
        else:
            return None

    df_binary['Label'] = df_binary['Score'].apply(label_score)

    df_binary = df_binary.dropna(subset=['Label'])

    df_binary.reset_index(drop=True, inplace=True)
    return df_binary
