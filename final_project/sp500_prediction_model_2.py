import pandas as pd
import yfinance as yf

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score 

import pathlib
import pickle

import mlflow
from prefect import flow, task
from prefect_aws import AwsCredentials
from prefect_aws.s3 import S3Bucket
from prefect.artifacts import create_markdown_artifact



@task(retries=3, retry_delay_seconds=2)
def prepare_data():
    sp500 = yf.Ticker('^GSPC').history('max')
    del sp500['Dividends']
    del sp500['Stock Splits']
    sp500['Tomorrow'] = sp500['Close'].shift(-1)
    sp500['Target'] = (sp500['Tomorrow'] > sp500['Close']).astype(int)
    sp500 = sp500.loc['1990-01-01':].copy()
    
    train = sp500.iloc[:-100]
    test = sp500.iloc[-100:]
    features = ['Close', 'Volume', 'Open', 'High', 'Low']

@task(log_prints=True)
def train_model():
    with mlflow.start_run():
        mlflow.autolog()
        
        mlflow.set_tag('developer', 'Timur')
        mlflow.set_tag('model', 'RandomForestClassifier')
        mlflow.set_tag('type', 'without_backtesting')
        
        mlflow.log_param('train-data-path', '/home/timur/work_hub/mlops_zoomcamp2023/Final_project/Data/train_data.csv')
        mlflow.log_param('test-data-path', '/home/timur/work_hub/mlops_zoomcamp2023/Final_project/Data/test_data.csv')
        
        n_estimators = 100
        min_samples_split = 100
        random_state = 1

        mlflow.log_param('n_estimators', n_estimators)
        mlflow.log_param('min_samples_split', min_samples_split)
        mlflow.log_param('random_state', random_state)
        
        model_1 = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, random_state=random_state)
        model_1.fit(train[features], train['Target'])
                        
        preds = model_1.predict(test[features])                 
        accuracy = precision_score(test['Target'], preds)
        mlflow.log_metric('accuracy', accuracy)
        
        with open('/home/timur/work_hub/mlops_zoomcamp2023/Final_project/Models/random_forest_model_1.pkl', 'wb') as f_out:
            pickle.dump(model_1, f_out)
            

@task(log_prints=True)
def predict(train, test, features, model):
    model.fit(train[features], train['Target'])
    preds = model.predict(test[features])
    preds = pd.Series(preds, index = test.index, name = 'Predictions')
    combined = pd.concat([test['Target'], preds], axis = 1)
    return combined

@task
def backtest(data, model, features, start = 2500, step = 250):
    all_predictors = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        # use def predict for generate predictions
        predictions = predict(train, test, features, model)
        all_predictors.append(predictions)
        
    return pd.concat(all_predictors)
predictions = backtest(sp500, model_1, features)

precision_score(predictions['Target'], predictions['Predictions'])










@flow
def main_flow(
    train_path: str = "/home/timur/work_hub/mlops_zoomcamp2023/Final_project/Data/train_data.csv",
    test_path: str = "/home/timur/work_hub/mlops_zoomcamp2023/Final_project/Data/test_data.csv",
) -> None:
    """The main training pipeline"""

    # MLflow settings
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("sp500-experiment")

    # Load
    s3_bucket_block = S3Bucket.load('s3-bucker-block')
    s3_bucket_block.download_folder_to_path(from_folder='data', to_folder='data')
    
    df_train = read_data(train_path)
    df_test = read_data(test_path)

    # Transform
    X_train, X_val, y_train, y_val, dv = add_features(df_train, df_val)

    # Train
    train_best_model(X_train, X_val, y_train, y_val, dv)


if __name__ == "__main__":
    main_flow()