import pandas as pd
from typing import List
import pymc as pm
import numpy as np
import pandas as pd
import scipy.io
import glob
import os


def read_dd_data(path: str) -> pd.DataFrame:
    """
    Read data from the given path and return a DataFrame.

    action (1=certain, 2=uncertain, 0=missing)
    condition (1=reward,2=loss)

    return: pd.DataFrame with columns: participant, choice, condition, event_prob, odds, rt, rcert
    """
    data_files = glob.glob(path + "/*.mat")
    all_participants = []

    for file in data_files:
        mat_data = scipy.io.loadmat(file)
        
        # Extract training data
        data_train = pd.DataFrame(mat_data['data_train'], columns=[col[0] for col in mat_data['data_labels'].flatten()])
        
        # rename columns
        data_train = data_train.rename(columns={'certOutcome': 'rcert',
                                                'uncOutcome': 'runcert',
                                                'action (1=certain, 2=uncertain, 0=missing)': 'choice',
                                                'p_cert': 'experimental_condition',
                                                'outcome prob.': 'event_prob',
                                                'condition (1=reward,2=loss)': 'condition',
                                                'RT': 'rt',
                                                'odds': 'odds'})


        # Add participant identifier
        data_train['participant'] = os.path.basename(file).replace('.mat', '')
        
        all_participants.append(data_train)

    # Combine all participants into a single DataFrame
    df = pd.concat(all_participants, ignore_index=True)

    # Remove trials where choice is missing (0)
    df = df[df['choice'] != 0]

    # drop experimental_condition column
    df = df.drop(columns=['experimental_condition'])

    # normalize probabilities to be between 0 and 1
    df['event_prob'] = df['event_prob'] / 100

    # make choice 0 be certain and 1 be uncertain
    df['choice'] = df['choice'] - 1

    return df