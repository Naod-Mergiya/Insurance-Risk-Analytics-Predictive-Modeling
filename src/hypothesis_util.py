import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind

# Metric calculation functions
def claim_frequency(df, group_col):
    """
    Calculate claim frequency for each group.
    """
    return df.groupby(group_col)['TotalClaims'].apply(lambda x: (x > 0).mean())

def claim_severity(df, group_col):
    """
    Calculate claim severity for each group.
    """
    return df[df['TotalClaims'] > 0].groupby(group_col)['TotalClaims'].mean()

def margin(df, group_col):
    """
    Calculate margin (TotalPremium - TotalClaims) for each group.
    """
    return df.groupby(group_col).apply(lambda x: (x['TotalPremium'] - x['TotalClaims']).mean())

# Statistical test functions
def chi2_test(df, group_col, outcome_col):
    """
    Perform a chi-squared test for independence between group_col and outcome_col (binary).
    Returns: chi2, p-value, dof, expected
    """
    contingency = pd.crosstab(df[group_col], df[outcome_col])
    return chi2_contingency(contingency)

def t_test(group_a, group_b):
    """
    Perform Welch's t-test between two numerical samples.
    Returns: t-statistic, p-value
    """
    return ttest_ind(group_a, group_b, equal_var=False)

def plot_group_metric(metric_series, title, ylabel):
    import matplotlib.pyplot as plt
    metric_series.plot(kind='bar', figsize=(10, 5))
    plt.title(title)
    plt.ylabel(ylabel)
    plt.show()