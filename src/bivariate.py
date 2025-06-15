import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import save_plot

def correlation_analysis(df, numerical_cols):
    """Generate correlation matrix for numerical columns."""
    numerical_cols = [col for col in numerical_cols if col in df.columns]
    if len(numerical_cols) > 1:
        correlation_matrix = df[numerical_cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Correlation Matrix of Numerical Features', fontsize=14, pad=20)
        plt.show
    else:
        print("Warning: Insufficient numerical columns for correlation matrix.")

def scatter_premium_claims_postalcode(df, output_dir='visualizations'):
    """Scatter plot of monthly TotalPremium vs TotalClaims by PostalCode."""
    if all(col in df.columns for col in ['TransactionMonth', 'TotalPremium', 'TotalClaims', 'PostalCode']):
        top_postalcodes = df['PostalCode'].value_counts().head(5).index
        df_subset = df[df['PostalCode'].isin(top_postalcodes)]
        monthly_trends = df_subset.groupby(['TransactionMonth', 'PostalCode']).agg({
            'TotalPremium': 'mean',
            'TotalClaims': 'mean'
        }).reset_index()

        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=monthly_trends, x='TotalPremium', y='TotalClaims', hue='PostalCode', size='TotalClaims', palette='deep', alpha=0.7)
        plt.title('Monthly Avg TotalPremium vs TotalClaims by PostalCode', fontsize=14, pad=20)
        plt.xlabel('Average Total Premium (Rand)', fontsize=12)
        plt.ylabel('Average Total Claims (Rand)', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        save_plot('scatter_premium_claims_postalcode.png', output_dir)

def monthly_trends_postalcode(df):
    """Line plot of monthly TotalPremium and TotalClaims by PostalCode."""
    if all(col in df.columns for col in ['TransactionMonth', 'TotalPremium', 'TotalClaims', 'PostalCode']):
        top_postalcodes = df['PostalCode'].value_counts().head(5).index
        df_subset = df[df['PostalCode'].isin(top_postalcodes)]
        monthly_trends = df_subset.groupby(['TransactionMonth', 'PostalCode']).agg({
            'TotalPremium': 'mean',
            'TotalClaims': 'mean'
        }).reset_index()

        plt.figure(figsize=(12, 8))
        for postalcode in top_postalcodes:
            subset = monthly_trends[monthly_trends['PostalCode'] == postalcode]
            plt.plot(subset['TransactionMonth'], subset['TotalPremium'], label=f'{postalcode} Premium')
            plt.plot(subset['TransactionMonth'], subset['TotalClaims'], linestyle='--', label=f'{postalcode} Claims')
        plt.title('Monthly Trends in TotalPremium and TotalClaims by PostalCode', fontsize=14, pad=20)
        plt.xlabel('Transaction Month', fontsize=12)
        plt.ylabel('Amount (Rand)', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show

def loss_ratio_by_province(df, output_dir='visualizations'):
    """Bar chart of Loss Ratio by Province."""
    if all(col in df.columns for col in ['TotalPremium', 'TotalClaims', 'Province']):
        df['LossRatio'] = df['TotalClaims'] / df['TotalPremium']
        loss_by_province = df.groupby('Province')['LossRatio'].mean().sort_values()
        plt.figure(figsize=(10, 6))
        sns.barplot(x=loss_by_province.index, y=loss_by_province.values, palette='magma')
        plt.title('Average Loss Ratio by Province', fontsize=14, pad=20)
        plt.xlabel('Province', fontsize=12)
        plt.ylabel('Loss Ratio', fontsize=12)
        plt.xticks(rotation=45)
        save_plot('loss_ratio_province.png', output_dir)
        return loss_by_province
    return None

def premium_by_covertype_province(df, output_dir='visualizations'):
    """Heatmap of TotalPremium by Province and CoverType."""
    if all(col in df.columns for col in ['TotalPremium', 'Province', 'CoverType']):
        pivot_premium = df.pivot_table(values='TotalPremium', index='Province', columns='CoverType', aggfunc='mean')
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_premium, annot=True, cmap='YlGnBu', fmt='.0f')
        plt.title('Average TotalPremium by Province and CoverType', fontsize=14, pad=20)
        plt.xlabel('Cover Type', fontsize=12)
        plt.ylabel('Province', fontsize=12)
        save_plot('premium_province_covertype.png', output_dir)

def make_by_province(df, output_dir='visualizations'):
    """Heatmap of top Vehicle Makes by Province."""
    if all(col in df.columns for col in ['Make', 'Province']):
        top_makes = df['Make'].value_counts().head(5).index
        df_make_subset = df[df['Make'].isin(top_makes)]
        pivot_make = df_make_subset.pivot_table(values='TotalClaims', index='Province', columns='Make', aggfunc='count', fill_value=0)
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_make, annot=True, cmap='Blues', fmt='d')
        plt.title('Count of Top Vehicle Makes by Province', fontsize=14, pad=20)
        plt.xlabel('Vehicle Make', fontsize=12)
        plt.ylabel('Province', fontsize=12)
        save_plot('make_province.png', output_dir)

def run_bivariate_analysis(df, numerical_cols, categorical_cols, output_dir='visualizations'):
    """Run complete bivariate and geographic analysis."""
    print("\nRunning Bivariate and Geographic Analysis...")
    numerical_cols = [col for col in numerical_cols if col in df.columns]
    categorical_cols = [col for col in categorical_cols if col in df.columns]

    # Bivariate/Multivariate
    correlation_analysis(df, numerical_cols)
    scatter_premium_claims_postalcode(df, output_dir)
    monthly_trends_postalcode(df)

    # Geographic Trends
    loss_by_province = loss_ratio_by_province(df, output_dir)
    premium_by_covertype_province(df, output_dir)
    make_by_province(df, output_dir)

    print("Bivariate and Geographic Analysis Summary:")
    print("Numerical Columns Analyzed:", numerical_cols)
    print("Categorical Columns Analyzed:", categorical_cols)
    if loss_by_province is not None:
        print("\nLoss Ratio by Province:")
        print(loss_by_province)