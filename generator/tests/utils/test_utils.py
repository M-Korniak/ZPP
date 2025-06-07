import gzip
import os
import pandas as pd
import pytest
from src.utils.utils import unpack_and_read

@pytest.fixture
def csv_file(tmp_path):
    path = tmp_path / "test.csv"
    data = "col1,col2\n1,2\n3,4"
    path.write_text(data, encoding="utf-8")
    yield path
    path.unlink(missing_ok=True)

@pytest.fixture
def csv_gz_file(tmp_path):
    path = tmp_path / "test.csv.gz"
    data = "col1,col2\n5,6\n7,8"
    with gzip.open(path, 'wt', encoding="utf-8") as f:
        f.write(data)
    yield path
    path.unlink(missing_ok=True)

def test_read_normal_csv(csv_file):
    df = unpack_and_read(str(csv_file))
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)
    assert list(df.columns) == ["col1", "col2"]
    assert df.iloc[0, 0] == 1

def test_read_gzipped_csv(csv_gz_file):
    df = unpack_and_read(str(csv_gz_file))
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)
    assert df.iloc[1, 1] == 8

def test_none_input_returns_empty_df():
    df = unpack_and_read(None)
    assert isinstance(df, pd.DataFrame)
    assert df.empty

def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        unpack_and_read("non_existent_file.csv.gz")
