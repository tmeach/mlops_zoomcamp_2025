#!/usr/bin/env python
# coding: utf-8

import sys 
import os
import pandas as pd
import pickle


def read_data(filename, categorical):
    s3_endpoint_url = os.getenv('S3_ENDPOINT_URL')
    
    if s3_endpoint_url:
        options = {
            'client_kwargs': {
                'endpoint_url': s3_endpoint_url
            }
        }
        df = pd.read_parquet(filename, storage_options=options)
    else:
        df = pd.read_parquet(filename)
    return df

def prepare_data(df, categorical):
    df['duration'] = df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
    df['duration'] = df['duration'].dt.total_seconds()/60
    
    df =  df[(df['duration'] >= 1) & (df['duration'] <=60)].copy()
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

def save_data(df, output_file, options):
    df.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
    )

def main(year, month):
    input_file = f's3://nyc-duration/integration_test/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f's3://nyc-duration/predicted_data/taxi_type=yellow_year={year:04d}_month={month:02d}.parquet'
    
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)
    
    categorical = ['PULocationID', 'DOLocationID']
    
    
    df = read_data(input_file, categorical)
    df = prepare_data(df, categorical)
    
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)
    
    print('predicted mean duration:', y_pred.mean())
    
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred 
    
    options = {
        'client_kwargs': {
            'endpoint_url': os.getenv('S3_ENDPOINT_URL', 'http://localhost:4566')
        }
    }
    
    save_data(df_result, output_file, options)
    
if __name__ == '__main__':
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    main(year, month)
    
    
    