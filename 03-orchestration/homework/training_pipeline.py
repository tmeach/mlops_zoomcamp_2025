import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer
import mlflow
import mlflow.sklearn

filename = '/Users/pitsuevt/work_main/learning/datatalks/mlops_zoomcamp_2025/03-orchestration/data/yellow_tripdata_2023-03.parquet'

mlflow.set_tracking_uri("http://localhost:5000")
print("MLflow URI:", mlflow.get_tracking_uri())

try:
    print("Проверка метаданных файла...")
    metadata = pq.ParquetFile(filename).metadata
    print(f"Формат: {metadata}")
    
    print("Чтение файла...")
    df = pd.read_parquet(filename)
    print(f"{df.shape[0]} записей загружено")
except Exception as e:
    print(f"Ошибка: {e}")

def read_dataframe(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df['duration'].dt.total_seconds() / 60

    df = df[(df['duration'] >= 1) & (df['duration'] <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df

def feature_vectorizing(df_upgrade):    
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']

    dv = DictVectorizer()
    train_dicts = df_upgrade[categorical + numerical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)

    target = 'duration'
    y_train = df_upgrade[target].values

    return X_train, y_train, dv

df_upgrade = read_dataframe(filename)
X_train, y_train, dv = feature_vectorizing(df_upgrade)

with mlflow.start_run():
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    mlflow.sklearn.log_model(lr, "model")

    # Регистрация модели, только если сервер запущен
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
    model_name = "taxi-duration-model"
    try:
        mlflow.register_model(model_uri, model_name)
        print(f"Модель зарегистрирована под именем: {model_name}")
    except Exception as e:
        print(f"Ошибка регистрации модели: {e}")