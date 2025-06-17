import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics for the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        
    Returns:
        pd.DataFrame: Summary statistics.
    """
    try:
        stats = df.describe()
        logger.info("Computed summary statistics")
        return stats
    except Exception as e:
        logger.error(f"Error computing summary statistics: {str(e)}")
        raise

def detect_negative_values(df: pd.DataFrame, columns: list) -> pd.Series:
    """
    Detect negative values in specified columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list): List of columns to check.
        
    Returns:
        pd.Series: Count of negative values per column.
    """
    try:
        negative_counts = (df[columns] < 0).sum()
        logger.info("Detected negative values")
        return negative_counts
    except Exception as e:
        logger.error(f"Error detecting negative values: {str(e)}")
        raise

def detect_missing_values(df: pd.DataFrame) -> tuple:
    """
    Detect missing values and their percentages.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        
    Returns:
        tuple: (missing_counts, missing_percentages)
    """
    try:
        missing_counts = df.isna().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        logger.info("Detected missing values")
        return missing_counts, missing_percentages
    except Exception as e:
        logger.error(f"Error detecting missing values: {str(e)}")
        raise

def sample_data (file_path,nrows):

    """
    Detect missing values and their percentages.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        
    Returns:
        tuple: (missing_counts, missing_percentages)
    """
    try:
        df_sample = pd.read_csv(file_path, nrows=5)
        print(df_sample.head())
        print("Columns:", df_sample.columns.tolist())
        logger.info("sample data")
        return sample_data
    except Exception as e:
        logger.error(f"Error sample data: {str(e)}")
        print("Error:", e)
    

