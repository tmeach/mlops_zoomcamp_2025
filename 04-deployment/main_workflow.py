#!/usr/bin/env python
# coding: utf-8

import os
import sys
import pickle
import pandas as pd


def read_data(filename, year, month):

    categorical = ['PULocationID', 'DOLocationID']
    try:
        df = pd.read_parquet(filename)
    except Exception as e:
        print(f'Error reading file {filename}:{e}')
        sys.exit(1)



    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df, categorical


def predict(df, categorical):
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    return y_pred


def create_output(df, y_pred, output_file):
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['prediction'] = y_pred

    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )


def run():
    if len(sys.argv) != 3:
        print("Usage: python script.py <year> <month>")
        sys.exit(1)


    year = int(sys.argv[1])
    month = int(sys.argv[2])

    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-0{month}.parquet'
    output_file = f'output/yellow_tripdata_{year}-0{month}.parquet'

    df, categorical = read_data(input_file, year, month)
    print(f'Successfully reading {input_file}')

    y_pred = predict(df, categorical)
    print(f'Successfully making predicitons\n')
    y_pred_mean = y_pred.mean()
    print(f'Mean duration - {y_pred_mean}')

    create_output(df, y_pred, output_file)
    print(f'Uploading data to {output_file}')


if __name__ == '__main__':
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)
    run()