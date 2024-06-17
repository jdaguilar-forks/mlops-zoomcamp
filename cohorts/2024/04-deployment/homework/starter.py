#!/usr/bin/env python
# coding: utf-8
# get_ipython().system('pip freeze | grep scikit-learn')
# get_ipython().system('python -V')
import pickle
import pandas as pd
import argparse


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


categorical = ['PULocationID', 'DOLocationID']


def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--year',
        type=int,
    )
    parser.add_argument(
        '--month',
        type=int,
    )

    # For testing.  Pass no arguments in production
    args = parser.parse_args()
    year = args.year
    month = args.month

    df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    print("Standard Deviation: ", y_pred.std())
    print("Mean: ", y_pred.mean())

    # Preparing output
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df['pred'] = y_pred
    
    output_file = f"yellow_tripdata_edited_{year:04d}-{month:02d}.parquet"

    df_result = df[['ride_id', 'pred']]

    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

