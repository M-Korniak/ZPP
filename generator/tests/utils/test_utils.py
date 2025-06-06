import gzip
import pandas as pd
import pytest
from unittest import mock
from src.utils.utils import unpack_and_read


@pytest.fixture
def sample_csv_gz(tmp_path):
    data = "col1,col2\n1,2\n3,4"
    file_path = tmp_path / "test.csv.gz"
    with gzip.open(file_path, 'wt') as f:
        f.write(data)
    return file_path


def test_unpack_and_read_from_file(sample_csv_gz):
    df = unpack_and_read(f_p=str(sample_csv_gz))
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)
    assert list(df.columns) == ['col1', 'col2']


def test_unpack_and_read_download(tmp_path):
    fake_data = "a,b\n5,6\n7,8"
    output_file = tmp_path / "data.csv.gz"

    def mock_download(url, output, quiet):
        with gzip.open(output, 'wt') as f:
            f.write(fake_data)
        return str(output)

    with mock.patch("gdown.download", side_effect=mock_download):
        with mock.patch("os.getcwd", return_value=str(tmp_path)):
            df = unpack_and_read(f_p=None)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)
    assert list(df.columns) == ['a', 'b']
