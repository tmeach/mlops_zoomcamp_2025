import pandas as pd
import yfinance as yf
import pickle
import mlflow
import boto3
from prefect_aws import AwsCredentials
from prefect_aws.s3 import S3Bucket
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score 


def load_data():
    try:
        session = boto3.Session(profile_name='aws_zoom')
        s3 = session.client('s3')
        
        sp500 = yf.Ticker('^GSPC').history('max')
        csv_buffer = sp500.to_csv(index=False)
        s3.put_object(Bucket='mlflow-artifacts-zoomcamp', Key='sp500_data.csv', Body=csv_buffer)
        return sp500
    
    except Exception as e:
        raise Exception(f"Error loading data: {e}")

def prepare_data(sp500):
    sp500 = sp500.copy()
    del sp500['Dividends']
    del sp500['Stock Splits']
    sp500['Tomorrow'] = sp500['Close'].shift(-1)
    sp500['Target'] = (sp500['Tomorrow'] > sp500['Close']).astype(int)
    sp500 = sp500.loc['1990-01-01':].copy()
    
    train = sp500.iloc[:-100]
    test = sp500.iloc[-100:]
    features = ['Close', 'Volume', 'Open', 'High', 'Low']
    
    return train, test, features

def predict(train, test, features, model):
    model.fit(train[features], train['Target'])
    preds = model.predict(test[features])
    preds = pd.Series(preds, index = test.index, name = 'Predictions')
    combined = pd.concat([test['Target'], preds], axis = 1)
    return combined


def backtest(data, model, features, start = 2500, step = 250):
    all_predictors = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        # use def predict for generate predictions
        predictions = predict(train, test, features, model)
        all_predictors.append(predictions)
        
    return pd.concat(all_predictors)

def train_model(train, features):
    with mlflow.start_run():
        mlflow.set_tag({'developer': 'Timur', 'model': 'RandomForestClassifier', 'type': 'with_backtesting'})
        
        n_estimators = 100
        min_samples_split = 100
        random_state = 1
    
        model = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, random_state=random_state)
        model.fit(train[features], train['Target'])
        
        predictions = backtest(train, model, features)                 
        accuracy = precision_score(predictions['Target'], predictions['Predictions'])
        mlflow.log_metric('accuracy', accuracy)
        
        # path to model locally and path to s3
        local_model_file = 'Models/random_forest_model.pkl'
        s3_model_path = 's3://mlflow-artifacts-zoomcamp/models/'
        
        #save model to s3
        with open(local_model_file, 'wb') as f_out:
            pickle.dump(model, f_out)
        s3.upload_file(local_model_file, S3_BUCKET, s3_model_path)


def main_flow():
    
    # MLflow settings
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("sp500-experiment")
    mlflow.autolog()
    
    # Load data
    sp500_data = load_data()
    
    # Transform data
    train_data, test_data, selected_features = prepare_data(sp500_data)
    
    # Train
    predictions = predict(train_data, test_data, selected_features, model)
    backtest_results = backtest(sp500_data, model, selected_features)
    trained_model = train_model(backtest_results, model, selected_features)

