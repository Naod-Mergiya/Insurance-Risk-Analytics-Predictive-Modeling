import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert 'TransactionMonth' column to datetime and extract hour into a new column.

    Args:
        df (pd.DataFrame): Input DataFrame with 'TransactionMonth' column.

    Returns:
        pd.DataFrame: DataFrame with 'TransactionMonth' as datetime and new 'Hour' column.
    """
    try:
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
        #df['Hour'] = df['TransactionMonth'].dt.hour
        logger.info("Converted TransactionMonth to datetime and extracted Hour.")
        return df
    except Exception as e:
        logger.error(f"Error converting timestamp: {str(e)}")
        raise



def impute_missing_values(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Impute missing values in specified columns with median.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list): List of columns to impute.
        
    Returns:
        pd.DataFrame: DataFrame with imputed values.
    """
    try:
        for col in columns:
            if df[col].isna().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                logger.info(f"Imputed {df[col].isna().sum()} missing values in {col} with median: {median_val}")
        return df
    except Exception as e:
        logger.error(f"Error imputing missing values: {str(e)}")
        raise