import pandas as pd
from datetime import datetime

from batch import prepare_data


def dt(hour, minute, second=0):
    """Helper function to create datetime object."""
    return datetime(2023, 1, 1, hour, minute, second)


def test_prepare_data():

    input_data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]
    input_columns = [
        "PULocationID",
        "DOLocationID",
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
    ]
    categorical = ["PULocationID", "DOLocationID"]
    input_df = pd.DataFrame(input_data, columns=input_columns)

    output_df = prepare_data(input_df, categorical)

    data_expected = [
        ("-1", "-1", 9),
        ("1", "1", 8),
    ]

    columns_test = ["PULocationID", "DOLocationID", "duration"]
    expected_df = pd.DataFrame(data_expected, columns=columns_test)

    print(input_df)

    assert (output_df["PULocationID"] == expected_df["DOLocationID"]).all()
    assert (output_df["DOLocationID"] == expected_df["PULocationID"]).all()
    assert (output_df["duration"] == expected_df["duration"]).all()
