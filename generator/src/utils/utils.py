import os
import gzip
import pandas as pd

def unpack_and_read(f_p: str = None) -> pd.DataFrame:
    """
    Reads a gzipped CSV file into a Pandas DataFrame.
    If `f_p` is None, returns an empty DataFrame.

    Args:
        f_p (str): Path to the .csv.gz file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    if f_p is None:
        return pd.DataFrame()

    if not os.path.exists(f_p):
        raise FileNotFoundError(f"File not found: {f_p}")

    if f_p.endswith('.gz'):
        with gzip.open(f_p, 'rt', encoding='utf-8') as f:
            return pd.read_csv(f)
    else:
        return pd.read_csv(f_p)
