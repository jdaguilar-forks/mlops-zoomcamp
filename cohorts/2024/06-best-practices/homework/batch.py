#!/usr/bin/env python
# coding: utf-8
import os
import pickle
import sys

import pandas as pd


def get_input_path(year, month):
    default_input_pattern = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet"
    input_pattern = os.getenv("INPUT_FILE_PATTERN", default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    default_output_pattern = "s3://nyc-duration-prediction-alexey/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet"
    output_pattern = os.getenv("OUTPUT_FILE_PATTERN", default_output_pattern)
    return output_pattern.format(year=year, month=month)


def prepare_data(df, categorical):
    """
    Convert dataframe to list of dicts for encoding and prediction.

    Parameters:
    df (pd.DataFrame): Dataframe containing pickup and dropoff times.
    categorical (list[str]): List of categorical columns.

    Returns:
    pd.DataFrame: Dataframe with encoded data.
    """
    # Extract duration in minutes
    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60

    # Filter out rows where duration is outside the range of 1 to 60 minutes
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    # Convert categorical columns to string type for encoding
    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")

    return df


def save_data(filename, df):
    """
    Save dataframe to parquet file in S3

    Parameters:
    filename (str): Path of the file to be saved.
    df (pd.DataFrame): Dataframe to be saved in parquet format.
    """
    S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
    options = None

    if S3_ENDPOINT_URL is not None:
        options = {"client_kwargs": {"endpoint_url": S3_ENDPOINT_URL}}

    df.to_parquet(filename, engine="pyarrow", index=False, storage_options=options)


def read_data(filename, categorical):
    """
    Read data from parquet file and prepare it for prediction.

    Parameters:
    filename (str): Path to parquet file.
    categorical (list[str]): List of categorical columns.

    Returns:
    pd.DataFrame: Dataframe with prepared data for prediction.
    """
    S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
    options = None

    if S3_ENDPOINT_URL is not None:
        options = {"client_kwargs": {"endpoint_url": S3_ENDPOINT_URL}}

    df = pd.read_parquet(filename, storage_options=options)

    return prepare_data(df, categorical)


def main(year, month):
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)

    with open("model.bin", "rb") as f_in:
        dv, lr = pickle.load(f_in)

    categorical = ["PULocationID", "DOLocationID"]

    df = read_data(input_file, categorical)
    df["ride_id"] = f"{year:04d}/{month:02d}_" + df.index.astype("str")

    dicts = df[categorical].to_dict(orient="records")
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print("predicted mean duration:", y_pred.mean())

    df_result = pd.DataFrame()
    df_result["ride_id"] = df["ride_id"]
    df_result["predicted_duration"] = y_pred

    save_data(output_file, df_result)


if __name__ == "__main__":
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    main(year, month)
