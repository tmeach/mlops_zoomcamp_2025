from moto import mock_s3
import boto3
import pandas as pd
from sp500_prediction import load_data  # Замените на путь к вашему модулю

@mock_s3
def test_load_data():
    # Создание тестового бакета S3
    s3 = boto3.client('s3', region_name='us-east-1')
    s3.create_bucket(Bucket='mlflow-artifacts-zoomcamp')
    
    # Мокирование профиля AWS
    aws_profile = 'aws_zoom'
    session = boto3.Session(profile_name=aws_profile)
    boto3.setup_default_session(region_name='us-east-1')
    
    # Вызов функции load_data
    sp500 = load_data()
    
    # Проверки
    assert sp500 is not None
    assert isinstance(sp500, pd.DataFrame)