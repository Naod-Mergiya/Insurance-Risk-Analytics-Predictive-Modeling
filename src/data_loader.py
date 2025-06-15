import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        
        df = pd.read_csv(file_path, delimiter='|', encoding='utf-8', low_memory=False)
        logger.info(f"Successfully loaded data from {file_path} with {len(df)} rows")
        
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