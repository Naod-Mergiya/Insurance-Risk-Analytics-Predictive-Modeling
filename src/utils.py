import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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