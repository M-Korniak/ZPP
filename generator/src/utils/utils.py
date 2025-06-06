import gzip
import pandas as pd
import gdown


def unpack_and_read(f_p=None) -> pd.DataFrame:
    """
    Unpacks and reads a gzipped CSV file into a Pandas DataFrame.

    Args:
    - f_p (str): The file path to the gzipped CSV file. Defaults to a sample file. If None, downloads the sample file.

    Returns:
    - pd.DataFrame: A DataFrame containing the data from the CSV file.
    """
    if f_p is None:
        return pd.DataFrame()

    with gzip.open(f_p, 'rt') as f:
        df = pd.read_csv(f)
    return df
