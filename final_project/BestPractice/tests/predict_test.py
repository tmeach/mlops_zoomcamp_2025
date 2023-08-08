import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sp500_prediction import predict 
import pytest

@pytest.fixture
def test_data():
    np.random.seed(42)
    n_samples = 100
    features = ['Feature1', 'Feature2', 'Feature3', 'Feature4']
    data = {
        'Feature1': np.random.rand(n_samples),
        'Feature2': np.random.rand(n_samples),
        'Feature3': np.random.rand(n_samples),
        'Feature4': np.random.rand(n_samples),
        'Target': np.random.randint(0, 2, n_samples)
    }
    return pd.DataFrame(data), features

def test_predict(test_data):
    train_data, features = test_data
    test_data = train_data.copy()

    model = RandomForestClassifier()
    result = predict(train_data, test_data, features, model)
    
    assert isinstance(result, pd.DataFrame)
    assert 'Target' in result.columns
    assert 'Predictions' in result.columns
    assert len(result) == len(test_data)
    assert all(result['Predictions'] >= 0) and all(result['Predictions'] <= 1)
