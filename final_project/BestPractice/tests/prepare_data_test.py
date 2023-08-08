import pandas as pd
from sp500_prediction import prepare_data 

def test_prepare_data():
    # Создание тестовых данных
    data = pd.DataFrame({
        'Date': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'Close': [100, 110, 105],
        'Dividends': [0.5, 0.2, 0.3],
        'Stock Splits': [2, 1, 1]
    })
    
    # Вызов функции prepare_data
    train, test, features = prepare_data(data)
    
    # Проверки
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    assert isinstance(features, list)
    
    assert 'Dividends' not in train.columns
    assert 'Stock Splits' not in train.columns
    
    assert 'Tomorrow' in test.columns
    assert 'Target' in test.columns
    assert 'Close' in features
    assert 'Volume' in features
    assert 'Open' in features
    assert 'High' in features
    assert 'Low' in features