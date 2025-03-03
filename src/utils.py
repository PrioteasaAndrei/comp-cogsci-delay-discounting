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
        unique_participants = df['participant'].unique()[:30]
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

    """
    test_indices = []

    for participant in df['participant'].unique():
        participant_data = df[df['participant'] == participant]
        n_trials = len(participant_data)
        n_test = int(test_size * n_trials) 
        
        participant_test_indices = participant_data.iloc[-n_test:].index
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



def normal_to_lognormal_std(mu_Y, sigma_Y):
    """
    Convert normal standard deviation to log-normal standard deviation with numerical stability.
    
    Parameters:
    - mu_Y: Mean of the underlying normal distribution.
    - sigma_Y: Standard deviation of the underlying normal distribution.
    
    Returns:
    - sigma_X: Standard deviation of the log-normal distribution.
    """
    # Clip values to prevent overflow
    mu_Y = np.clip(mu_Y, -100, 100)
    sigma_Y = np.clip(sigma_Y, 0, 10)  # standard deviation should be positive
    
    try:
        # Use log1p for numerical stability when computing exp(sigma_Y^2) - 1
        variance_term = np.log1p(np.exp(np.minimum(sigma_Y**2, 100)) - 1)
        
        # Compute mu_X with clipping
        exp_term = np.clip(mu_Y + (sigma_Y**2) / 2, -100, 100)
        mu_X = np.exp(exp_term)
        
        # Compute final sigma
        sigma_X = mu_X * np.sqrt(np.exp(variance_term))
        
        return sigma_X
    except Exception as e:
        print(f"Error in conversion: mu_Y={mu_Y}, sigma_Y={sigma_Y}")
        print(f"Exception: {e}")
        return np.nan

def create_comprehensive_validation_split(df, participant_holdout_ratio=0.2, trial_holdout_ratio=0.2, random_seed=42):
    """
    Create a comprehensive validation split that tests both generalization to new participants
    and generalization to new trials from existing participants.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The full dataset
    participant_holdout_ratio : float, default=0.2
        Proportion of participants to completely hold out for testing
    trial_holdout_ratio : float, default=0.2
        Proportion of trials to hold out from participants that are included in training
    random_seed : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'train_set': DataFrame with training data (subset of participants and subset of trials)
        - 'test_set_new_participants': DataFrame with data from completely new participants
        - 'test_set_new_trials': DataFrame with new trials from participants in the training set
        - 'test_set_combined': DataFrame combining both test sets
    """
    np.random.seed(random_seed)
    
    # Get unique participants
    unique_participants = df['participant_idx'].unique()
    n_participants = len(unique_participants)
    
    # Determine how many participants to hold out completely
    n_holdout_participants = max(1, int(n_participants * participant_holdout_ratio))
    
    # Randomly select participants to hold out completely
    holdout_participants = np.random.choice(
        unique_participants, 
        size=n_holdout_participants, 
        replace=False
    )
    
    # Participants to include in training (with some trials held out)
    training_participants = np.setdiff1d(unique_participants, holdout_participants)
    
    # Create test set of completely new participants
    test_set_new_participants = df[df['participant_idx'].isin(holdout_participants)].copy()
    
    # Initialize DataFrames for training set and test set of new trials
    train_set_parts = []
    test_set_new_trials_parts = []
    
    # For each training participant, split their trials
    for participant in training_participants:
        participant_data = df[df['participant_idx'] == participant].copy()
        
        # Shuffle the data
        participant_data = participant_data.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        
        # Calculate split point
        n_trials = len(participant_data)
        n_test_trials = max(1, int(n_trials * trial_holdout_ratio))
        
        # Split the data
        train_trials = participant_data.iloc[:-n_test_trials].copy()
        test_trials = participant_data.iloc[-n_test_trials:].copy()
        
        # Add to the respective sets
        train_set_parts.append(train_trials)
        test_set_new_trials_parts.append(test_trials)
    
    # Combine the parts
    train_set = pd.concat(train_set_parts).reset_index(drop=True)
    test_set_new_trials = pd.concat(test_set_new_trials_parts).reset_index(drop=True)
    
    # Create a combined test set
    test_set_combined = pd.concat([test_set_new_participants, test_set_new_trials]).reset_index(drop=True)
    
    return {
        'train_set': train_set,
        'test_set_new_participants': test_set_new_participants.reset_index(drop=True),
        'test_set_new_trials': test_set_new_trials,
        'test_set_combined': test_set_combined
    }


def create_k_fold_comprehensive_validation(df, k=5, participant_holdout_ratio=0.2, trial_holdout_ratio=0.2):
    """
    Create k different comprehensive validation splits for robust cross-validation.
    Each fold has different participants held out and different train/test splits.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The full dataset
    k : int, default=5
        Number of folds to create
    participant_holdout_ratio : float, default=0.2
        Proportion of participants to completely hold out for testing in each fold
    trial_holdout_ratio : float, default=0.2
        Proportion of trials to hold out from participants included in training
    
    Returns:
    --------
    list
        List of dictionaries, each containing train/test splits for one fold
    """
    all_folds = []
    
    for fold in range(k):
        # Create split with a different random seed for each fold
        split = create_comprehensive_validation_split(
            df, 
            participant_holdout_ratio=participant_holdout_ratio,
            trial_holdout_ratio=trial_holdout_ratio,
            random_seed=42+fold  # Different seed for each fold
        )
        
        # Add fold index to the split dictionary
        split['fold'] = fold
        all_folds.append(split)
    
    return all_folds

def interpret_beta2_significance(results):
    """
    Provides a comprehensive interpretation of the test_beta2_significance results.
    
    Parameters:
    -----------
    results : dict
        Dictionary returned by test_beta2_significance, containing:
        - 'group_level_significant': Boolean indicating if group-level β₂ is significant
        - 'individual_significant_proportion': Proportion of participants with significant β₂
        - 'beta2_mean': Mean of the group-level β₂ parameter
        - 'beta2_hdi': 95% HDI interval for group-level β₂
        
    Returns:
    --------
    dict
        Dictionary containing interpretation and recommendations
    """
    # Extract key metrics
    group_significant = results['group_level_significant']
    indiv_proportion = results['individual_significant_proportion']
    beta2_mean = results['beta2_mean']
    beta2_hdi = results['beta2_hdi']
    zero_in_hdi = (beta2_hdi[0] <= 0 <= beta2_hdi[1])
    
    # Determine evidence strength
    if group_significant and indiv_proportion > 0.5:
        evidence = "strong evidence"
    elif group_significant or indiv_proportion > 0.3:
        evidence = "moderate evidence"
    elif indiv_proportion > 0.1:
        evidence = "weak evidence"
    else:
        evidence = "no significant evidence"
    
    # Determine effect direction if significant
    if abs(beta2_mean) > 0.01 and not zero_in_hdi:
        if beta2_mean > 0:
            effect = ("positive quadratic effect, suggesting reaction times increase for "
                     "moderately difficult decisions (inverted U-shape)")
        else:
            effect = ("negative quadratic effect, suggesting reaction times decrease for "
                     "moderately difficult decisions (U-shape)")
    else:
        effect = "minimal quadratic effect that is practically indistinguishable from zero"
    
    # Generate summary interpretation
    if group_significant:
        summary = (f"The analysis shows {evidence} that the quadratic term (β₂) significantly "
                  f"improves your reaction time model. There is a {effect}.")
    else:
        summary = (f"The analysis shows {evidence} that the quadratic term (β₂) improves "
                  f"your reaction time model. There is a {effect}.")
    
    # Generate detailed explanation
    details = []
    details.append(f"Group-level β₂ mean: {beta2_mean:.4f}")
    details.append(f"Group-level β₂ 95% HDI: [{beta2_hdi[0]:.4f}, {beta2_hdi[1]:.4f}]")
    
    if zero_in_hdi:
        details.append("The 95% HDI interval contains zero, meaning we cannot reject the null hypothesis that β₂ = 0 at the group level.")
    else:
        details.append("The 95% HDI interval excludes zero, providing evidence against the null hypothesis that β₂ = 0 at the group level.")
    
    if indiv_proportion < 0.2:
        details.append(f"Only {indiv_proportion*100:.1f}% of participants showed a significant individual effect, suggesting the quadratic pattern is rare in your sample.")
    else:
        details.append(f"{indiv_proportion*100:.1f}% of participants showed a significant individual effect.")
    
    # Generate practical recommendations
    if group_significant or indiv_proportion > 0.3:
        recommendations = [
            "Include the quadratic term in your final model",
            "Examine individual differences in the quadratic effect",
            "Consider how this nonlinear effect aligns with theoretical expectations"
        ]
    else:
        recommendations = [
            "The linear model is likely sufficient; added complexity of the quadratic model is not justified",
            "Reaction times appear to change approximately linearly with decision difficulty",
            "Consider exploring alternative predictors or model formulations"
        ]
    
    if 0.1 < indiv_proportion < 0.3:
        recommendations.append("Investigate participants with significant quadratic effects separately - they may have different decision strategies")
    
    # Compile complete interpretation
    interpretation = {
        "summary": summary,
        "details": details,
        "recommendations": recommendations,
        "evidence_strength": evidence,
        "effect_direction": "neutral" if zero_in_hdi else ("positive" if beta2_mean > 0 else "negative"),
        "model_selection": "quadratic" if (group_significant or indiv_proportion > 0.3) else "linear"
    }
    
    return interpretation


def print_beta2_interpretation(results):
    """
    Prints a nicely formatted interpretation of the test_beta2_significance results.
    
    Parameters:
    -----------
    results : dict
        Dictionary returned by test_beta2_significance
    """
    interp = interpret_beta2_significance(results)
    
    print("\n" + "="*80)
    print(" QUADRATIC TERM (β₂) SIGNIFICANCE ANALYSIS ".center(80, "="))
    print("="*80 + "\n")
    
    print("SUMMARY:")
    print(interp["summary"])
    print()
    
    print("DETAILED ANALYSIS:")
    for detail in interp["details"]:
        print(f"• {detail}")
    print()
    
    print("RECOMMENDATIONS:")
    for i, rec in enumerate(interp["recommendations"], 1):
        print(f"{i}. {rec}")
    print()
    
    print("MODEL SELECTION:")
    if interp["model_selection"] == "quadratic":
        print("✓ The quadratic model is supported by the data")
    else:
        print("✓ The linear model is sufficient; quadratic term not justified")
    
    print("\n" + "="*80 + "\n")

def test_beta2_significance(trace):
    """
    Test if the quadratic parameter (beta2) is significantly different from zero
    by examining posterior distributions.
    """
    # Extract β₂ posterior samples (both group-level and individual-level)
    mu_beta2 = trace.posterior["mu_beta2"].values.flatten()
    individual_beta2 = trace.posterior["beta2"].values
    
    # Group-level analysis
    beta2_mean = np.mean(mu_beta2)
    beta2_hdi = az.hdi(mu_beta2, hdi_prob=0.95)
    zero_in_hdi = (beta2_hdi[0] <= 0 <= beta2_hdi[1])
    
    # Individual-level analysis
    n_participants = individual_beta2.shape[2]
    significant_participants = 0
    
    for p in range(n_participants):
        participant_beta2 = individual_beta2[:, :, p].flatten()
        p_hdi = az.hdi(participant_beta2, hdi_prob=0.95)
        if not (p_hdi[0] <= 0 <= p_hdi[1]):
            significant_participants += 1
    
    proportion_significant = significant_participants / n_participants
    
    # Print results
    print(f"Group-level β₂ mean: {beta2_mean:.4f}")
    print(f"Group-level β₂ 95% HDI: [{beta2_hdi[0]:.4f}, {beta2_hdi[1]:.4f}]")
    print(f"Zero within HDI: {zero_in_hdi}")
    print(f"Participants with significant β₂: {significant_participants}/{n_participants} ({proportion_significant*100:.1f}%)")
    
    return {
        "group_level_significant": not zero_in_hdi,
        "individual_significant_proportion": proportion_significant,
        "beta2_mean": beta2_mean,
        "beta2_hdi": beta2_hdi
    }