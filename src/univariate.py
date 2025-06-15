import seaborn as sns
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def univariate_numerical_analysis(df, numerical_cols):
    """Perform univariate analysis for numerical columns with histograms and box plots."""
    for col in numerical_cols:
        # Histogram with KDE
        plt.figure()
        sns.histplot(df[col].dropna(), bins=30, kde=True, color='skyblue')
        plt.title(f'Distribution of {col}', fontsize=14, pad=20)
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.show

        # Box plot for outlier detection
        plt.figure()
        sns.boxplot(data=df, y=col, color='lightcoral')
        plt.title(f'Box Plot of {col} (Outlier Detection)', fontsize=14, pad=20)
        plt.ylabel(col, fontsize=12)
        plt.show

def univariate_categorical_analysis(df, categorical_cols):
    """Perform univariate analysis for categorical columns with bar charts."""
    for col in categorical_cols:
        plt.figure()
        sns.countplot(data=df, x=col, palette='viridis')
        plt.title(f'Frequency of {col}', fontsize=14, pad=20)
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45)
        plt.show

def run_univariate_analysis(df, numerical_cols, categorical_cols):
    """Run complete univariate analysis."""
    print("\nRunning Univariate Analysis...")
    numerical_cols = [col for col in numerical_cols if col in df.columns]
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    
    if numerical_cols:
        univariate_numerical_analysis(df, numerical_cols)
    else:
        print("Warning: No numerical columns found for univariate analysis.")
    
    if categorical_cols:
        univariate_categorical_analysis(df, categorical_cols)
    else:
        print("Warning: No categorical columns found for univariate analysis.")
    
    print("Univariate Analysis Summary:")
    print("Numerical Columns Analyzed:", numerical_cols)
    print("Categorical Columns Analyzed:", categorical_cols)