import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Set Seaborn style

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def load_data(file_path: str) -> pd.DataFrame:
         """
         Load CSV data from the specified file path.
         
         Args:
             file_path (str): Path to the CSV file.
             
         Returns:
             pd.DataFrame: Loaded DataFrame.
             
         Raises:
             FileNotFoundError: If the file does not exist.
             pd.errors.EmptyDataError: If the file is empty.
         """
         try:
             if not os.path.exists(file_path):
                 raise FileNotFoundError(f"File not found: {file_path}")
             
             df = pd.read_csv(file_path, delimiter='|', encoding='utf-8')
             logger.info(f"Successfully loaded data from {file_path} with {len(df)} rows")
             
             # Basic validation
             if df.empty:
                 raise pd.errors.EmptyDataError("The CSV file is empty")
             
             print("Available Columns:")
             print(df.columns.tolist())
             print("\nData Types:")
             print(df.dtypes)
             return df
         except Exception as e:
             logger.error(f"Error loading data: {e}")
             raise

def clean_numerical_columns(df, numerical_cols):
    """Clean numerical columns by handling comma-separated formats."""
    for col in numerical_cols:
        if col in df.columns and df[col].dtype == 'object':
            try:
                df[col] = df[col].str.replace(',', '.', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                print(f"Converted {col} to numeric.")
            except Exception as e:
                print(f"Error converting {col} to numeric: {e}")
    return df

def convert_datetime(df, date_col='TransactionMonth'):
    """Convert date column to datetime."""
    if date_col in df.columns:
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            print(f"\n{date_col} converted to datetime.")
        except Exception as e:
            print(f"Error converting {date_col} to datetime: {e}")
    return df

def save_plot(filename, output_dir='visualizations'):
    """Save plot to PNG in output directory."""
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
    plt.close()

def filter_columns(df, possible_cols):
    """Filter columns that exist in the dataframe."""
    return [col for col in possible_cols if col in df.columns]

def save_dataframe(df: pd.DataFrame, output_path: str) -> None:
    """
    Save DataFrame to a CSV file.
    
    Args:
        df (pd.DataFrame): DataFrame to save.
        output_path (str): Path to save the CSV file.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved DataFrame to {output_path}")
    except Exception as e:
        logger.error(f"Error saving DataFrame: {str(e)}")
        raise