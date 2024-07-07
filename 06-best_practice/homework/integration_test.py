import os
import pandas as pd
import boto3
import pyarrow.parquet as pq
import io

def run_batch_script():
    os.system('python batch_flow.py 2023 1')

def read_result_from_s3():
    s3_endpoint_url = os.getenv('S3_ENDPOINT_URL', 'http://localhost:4566')
    s3 = boto3.client('s3', endpoint_url=s3_endpoint_url)

    bucket_name = 'nyc-duration'
    file_name = 'predicted_data/taxi_type=yellow_year=2023_month=01.parquet'

    obj = s3.get_object(Bucket=bucket_name, Key=file_name)
    df = pq.read_table(io.BytesIO(obj['Body'].read())).to_pandas()
    return df

def verify_result(df):
    assert 'ride_id' in df.columns
    assert 'predicted_duration' in df.columns
    print('Verification passed!')

if __name__ == '__main__':
    run_batch_script()
    result_df = read_result_from_s3()
    verify_result(result_df)
