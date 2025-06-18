#!/usr/bin/env python
# coding: utf-8

import os
import sys
import pickle
import pandas as pd



taxi_type = sys.argv[1]
year = sys.argv[2]
month = sys.argv[3]


input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year}-{month}.parquet'
output_file = 'output/{taxi_type}_tripdata_{year}-{month}.parquet'


with open ('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df


df = read_data(input)


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)

print(y_pred.mean())

df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

df_result = pd.DataFrame()


df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)


if __name__ == '__main__':
    run()