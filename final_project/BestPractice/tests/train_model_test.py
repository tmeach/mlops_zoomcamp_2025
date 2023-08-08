import pytest
from unittest.mock import Mock
from sp500_prediction import train_model

@pytest.fixture
def mock_model(mocker):
    return mocker.Mock()

@pytest.fixture
def mock_mlflow(mocker):
    return mocker.patch('sp500_prediction.mlflow')

@pytest.fixture
def mock_backtest(mocker):
    return mocker.patch('sp500_prediction.backtest')

@pytest.fixture
def mock_pickle(mocker):
    return mocker.patch('sp500_prediction.pickle')

@pytest.fixture
def mock_s3(mocker):
    return mocker.patch('sp500_prediction.s3')

def test_train_model(mock_model, mock_mlflow, mock_backtest, mock_pickle, mock_s3):
    train_data = Mock()
    features = ['feature1', 'feature2']
    
    mock_model.fit.return_value = None
    mock_backtest.return_value = Mock()
    
    train_model(train_data, features)
    
    assert mock_model.fit.called
    assert mock_backtest.called
    assert mock_mlflow.log_metric.called
    assert mock_pickle.dump.called
    assert mock_s3.upload_file.called
