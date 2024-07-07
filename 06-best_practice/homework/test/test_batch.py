import pytest
import pandas as pd
from datetime import datetime

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def prepare_data(df, categorical):
    df['duration'] = df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
    df['duration'] = df['duration'].dt.total_seconds() / 60

    # Фильтрация по длительности поездки
    df = df[(df['duration'] >= 1) & (df['duration'] <= 60)].copy()

    # Преобразование столбцов categorical в строковый тип
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def test_prepare():
    # Test data
    test_data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    test_df = pd.DataFrame(test_data, columns=columns)

    # Expected data
    expected_data = [
        (-1, -1, dt(1, 1), dt(1, 10), 9.0),
        (1, 1, dt(1, 2), dt(1, 10), 8.0),
    ]

    expected_columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'duration']
    expected_df = pd.DataFrame(expected_data, columns=expected_columns)
    expected_df['PULocationID'] = expected_df['PULocationID'].astype(str)
    expected_df['DOLocationID'] = expected_df['DOLocationID'].astype(str)

    # Run function
    categorical = ['PULocationID', 'DOLocationID']
    result_df = prepare_data(test_df, categorical)

    # Assert the result
    pd.testing.assert_frame_equal(result_df, expected_df, check_dtype=True)



if __name__ == "__main__":
    pytest.main([__file__])
