import pandas as pd
from pathlib import Path


def load_data(data_dir: str) -> pd.DataFrame:
    """
    Load and merge monthly CSV files into one DataFrame.

    Parameters:
    -----------
    data_dir : str
        Path to the directory containing monthly CSV files.

    Returns:
    --------
    pd.DataFrame
        A single DataFrame containing all monthly data.
    """

    try:
        data_path = Path(data_dir)

        files = sorted(data_path.glob("*.csv"))

        if not files:
            raise FileNotFoundError(
                f"No CSV files found in: {data_dir}"
            )

        dataframes = []

        for file in files:
            print(f"Loading: {file.name}")  

            monthly_df = pd.read_csv(file)

            dataframes.append(monthly_df)

        df = pd.concat(
            dataframes,
            ignore_index=True
        )
    
        
        return df

    except Exception as e:
        print(f"Error loading data: {e}")
        
        return pd.DataFrame()
    
    
if __name__ == "__main__":
    data_directory = "data/"
    df = load_data(data_directory)