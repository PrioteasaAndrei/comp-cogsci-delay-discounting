import pandas as pd
from typing import List
import pymc as pm
import numpy as np
import pandas as pd
import scipy.io
import glob
import os
from typing import Tuple
import arviz as az
import matplotlib.pyplot as plt

def read_dd_data(path: str, standardize: bool = True, reduced_data: bool = False) -> pd.DataFrame:
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

    # Convert participant IDs to integer indices
    participants = df["participant"].unique()
    n_participants = len(participants)
    participant_idx = {pid: i for i, pid in enumerate(participants)}
    df["participant_idx"] = df["participant"].map(participant_idx)


    # Check for NaN values in all columns
    if df.isna().any().any():
        nan_columns = df.columns[df.isna().any()].tolist()
        raise ValueError(f"NaN values found in columns: {nan_columns}")
    
    if reduced_data:
        # Limit to 10 participants
        unique_participants = df['participant'].unique()[:10]
        df = df[df['participant'].isin(unique_participants)]
        
        # Limit to 100 trials per participant
        df = df.groupby('participant').head(100).reset_index(drop=True)

    if standardize:
        # used later for unscaling
        # need to log them so that they are normally distributed and can apply standardization
        df["rt"] = np.log(df["rt"])
        scaling_factors = {
            "rt_mean": df['rt'].mean(), "rt_std": df['rt'].std(),
            "rcert_mean": df['rcert'].mean(), "rcert_std": df['rcert'].std(),
            "runcert_mean": df['runcert'].mean(), "runcert_std": df['runcert'].std()
        }

        df["rcert"] = (df["rcert"] - scaling_factors["rcert_mean"]) / scaling_factors["rcert_std"]
        df["runcert"] = (df["runcert"] - scaling_factors["runcert_mean"]) / scaling_factors["runcert_std"]

        df["rt"] = (df["rt"] - scaling_factors["rt_mean"]) / scaling_factors["rt_std"]

        return df, scaling_factors

    return df, None




def split_train_test(df, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into training and test sets.

    TODO: not checked
    """
    # Create train/test split by dropping 20% of trials per participant
    test_indices = []

    for participant in df['participant'].unique():
        participant_data = df[df['participant'] == participant]
        n_trials = len(participant_data)
        n_test = int(test_size * n_trials)  # 20% of trials
        
        # Randomly sample indices for test set
        participant_test_indices = participant_data.sample(n=n_test).index
        test_indices.extend(participant_test_indices)

    # Create train and test dataframes
    df_test = df.loc[test_indices].copy()
    df_train = df.drop(test_indices).copy()

    print(f"Train set size: {len(df_train)} trials")
    print(f"Test set size: {len(df_test)} trials")
    print(f"Test set percentage: {(len(df_test) / len(df)) * 100:.1f}%")

    return df_train, df_test


def check_convergence_rhat(trace, threshold=1.05):
    """
    Check convergence of all parameters using R-hat values.
    
    Args:
        trace: ArviZ InferenceData object
        threshold: R-hat threshold for convergence warning (default: 1.05)
    """
    rhat_values = az.rhat(trace)
    
    # Define parameter groups for organized output
    param_groups = {
        "Group-level means": ["mu_k", "mu_beta", "mu_beta0", "mu_beta1", "mu_sigma_RT"],
        "Group-level standard deviations": ["sigma_k", "sigma_beta", "sigma_beta0", "sigma_beta1", "sigma_sigma_RT"],
        "Individual-level parameters": ["k", "beta", "beta0", "beta1", "sigma_RT"],
        "Deterministic variables": ["prob_choose_uncertain", "mu_RT"]
    }
    
    print("=== Convergence Analysis (R-hat) ===\n")
    
    all_good = True
    problematic_params = []
    
    for group_name, params in param_groups.items():
        print(f"\n{group_name}:")
        print("-" * (len(group_name) + 1))
        
        for param in params:
            if param in rhat_values.data_vars:
                values = rhat_values[param].values
                
                # For scalar parameters
                if np.size(values) == 1:
                    status = "✓" if values < threshold else "⚠️"
                    print(f"{status} {param}: {float(values):.3f}")
                    if values >= threshold:
                        all_good = False
                        problematic_params.append(f"{param} ({float(values):.3f})")
                
                # For vector parameters
                else:
                    max_rhat = np.max(values)
                    mean_rhat = np.mean(values)
                    status = "✓" if max_rhat < threshold else "⚠️"
                    print(f"{status} {param}:")
                    print(f"    mean: {mean_rhat:.3f}")
                    print(f"    max:  {max_rhat:.3f}")
                    print(f"    min:  {np.min(values):.3f}")
                    if max_rhat >= threshold:
                        all_good = False
                        problematic_params.append(f"{param} (max: {max_rhat:.3f})")
            else:
                print(f"⚪ {param}: Not found in trace")
    
    print("\n=== Summary ===")
    if all_good:
        print("✅ All parameters show good convergence (R-hat < 1.05)")
    else:
        print("❌ Convergence issues detected!")
        print("\nParameters requiring attention:")
        for param in problematic_params:
            print(f"  - {param}")




def plot_data_distributions(df):
    """
    Plot histograms of the raw data distributions.
    """
    # Create histograms of the raw data

    # Extract observed data
    rcert = df["rcert"].values
    runcert = df["runcert"].values
    participant_ids = df["participant_idx"].values  # Integer IDs
    event_prob = df["event_prob"].values
    choice_data = df["choice"].astype(int).values  # Ensure it's an integer for Bernoulli
    rt_data = df["rt"].values

    # Create histograms of the raw data
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Distribution of Raw Data')

    # Flatten axes for easier iteration
    axes = axes.flatten()

    # Plot histograms
    axes[0].hist(rcert, bins=30)
    axes[0].set_title('Certain Reward/Loss')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Count')

    axes[1].hist(runcert, bins=30)
    axes[1].set_title('Uncertain Reward/Loss')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Count')

    axes[2].hist(event_prob, bins=30)
    axes[2].set_title('Event Probability')
    axes[2].set_xlabel('Probability')
    axes[2].set_ylabel('Count')

    # Calculate proportions for binary variable
    unique_values, counts = np.unique(choice_data, return_counts=True)
    proportions = counts / len(choice_data)

    # Create bar plot for binary variable
    axes[3].bar(['Certain (0)', 'Uncertain (1)'], proportions)
    axes[3].set_title('Choice Distribution')
    axes[3].set_xlabel('Choice')
    axes[3].set_ylabel('Proportion')
    axes[3].set_ylim(0, 1)  # Set y-axis from 0 to 1 for proportions

    axes[4].hist(rt_data, bins=30)
    axes[4].set_title('Response Time')
    axes[4].set_xlabel('Time (ms)')
    axes[4].set_ylabel('Count')

    # Adjust layout to prevent overlap
    plt.tight_layout()


def split_data_k_fold(df: pd.DataFrame, k: int = 5) -> List[pd.DataFrame]:
    """
    Split the data into k-fold cross-validation sets.
    # TODO: think whether it makes sense to take entire participants or only some of their data
    """
    # Create k-fold cross-validation sets

    # TODO: random indexes can be used as we used a seed for the rng
    # TODO: splits should be roughly the same size
    

    return None