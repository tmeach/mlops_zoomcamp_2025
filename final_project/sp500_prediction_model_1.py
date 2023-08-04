import pandas as pd
import yfinance as yf
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score 
import mlflow
from prefect import flow, task

@task(retries=3, retry_delay_seconds=2)
def load_data():
    try:
        sp500 = yf.Ticker('^GSPC').history('max')
        sp500.to_csv('Data/sp500_data.csv')  # Используйте относительные пути
        return sp500
    except Exception as e:
        raise Exception(f"Error loading data: {e}")

@task(log_prints=True)
def prepare_data():
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

@task(log_prints=True)
def predict(train, test, features, model):
    model.fit(train[features], train['Target'])
    preds = model.predict(test[features])
    preds = pd.Series(preds, index = test.index, name = 'Predictions')
    combined = pd.concat([test['Target'], preds], axis = 1)
    return combined

@task(log_prints=True)
def backtest(data, model, features, start = 2500, step = 250):
    all_predictors = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        # use def predict for generate predictions
        predictions = predict(train, test, features, model)
        all_predictors.append(predictions)
        
    return pd.concat(all_predictors)

@task
def mlflow():
    with mlflow.start_run():
        mlflow.autolog()
        mlflow.set_tag({'developer': 'Timur', 'model': 'RandomForestClassifier', 'type': 'with_backtesting'})
        
        mlflow.log_param('train-data-path', 'Data/train_data.csv')
        mlflow.log_param('test-data-path', 'Data/test_data.csv')

        mlflow.log_param('n_estimators', n_estimators)
        mlflow.log_param('min_samples_split', min_samples_split)
        mlflow.log_param('random_state', random_state)
        
        n_estimators = 100
        min_samples_split = 100
        random_state = 1
        sp500 = sp500
        features = features
        model_1 = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, random_state=random_state)
        model_1.fit(train[features], train['Target'])
        
        predictions = backtest(sp500, model_1, features)                 
        accuracy = precision_score(predictions['Target'], predictions['Predictions'])
        mlflow.log_metric('accuracy', accuracy)

        with open('Models/random_forest_model_1.pkl', 'wb') as f_out:
            pickle.dump(model_1, f_out)

@flow
def main_flow():
    
    # MLflow settings
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("sp500-experiment")
    
    # Load data
    sp500_data = load_data()
    
    # Transform data
    train_data, test_data, selected_features = prepare_data(sp500_data)
    
    # Train
    model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
    predictions = predict(train_data, test_data, selected_features, model)
    backtest_results = backtest(sp500_data, model, selected_features)
    trained_model = mlflow(backtest_results, model, selected_features)

if __name__ == "__main__":
    main_flow()