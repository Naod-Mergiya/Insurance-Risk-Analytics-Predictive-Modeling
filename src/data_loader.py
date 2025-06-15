import pandas as pd
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
        
        df = pd.read_csv(file_path, delimiter='|' ,encoding='utf-8')
        logger.info(f"Successfully loaded data from {file_path} with {len(df)} rows")
        
        # Basic validation
        if df.empty:
            raise pd.errors.EmptyDataError("The CSV file is empty")
        
        return df
    
    except FileNotFoundError as e:
        logger.error(str(e))
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(str(e))
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise