"""Модуль для работы с AWS S3. Этот модуль предоставляет функции для работы с AWS S3, такие как создание и удаление ведер, загрузка и скачивание файлов и т.д.
"""

from time import sleep
from prefect_aws import AwsCredentials
from prefect_aws.s3 import S3Bucket

def create_aws_creds_block():
    my_aws_creds_obj = AwsCredentials(
        aws_access_key_id="AKIAYJXWGPR7Q27Y3VJ5", aws_secret_access_key="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"
    )
    my_aws_creds_obj.save(name="my-aws-creds", overwrite=True)


def create_s3_bucket_block():
    aws_creds = AwsCredentials.load("my-aws-creds")
    my_s3_bucket_obj = S3Bucket(
        bucket_name="my-first-bucket-abc", aws_credentials=aws_creds
    )
    my_s3_bucket_obj.save(name="s3-bucket-example", overwrite=True)


if __name__ == "__main__":
    create_aws_creds_block()
    sleep(5)
    create_s3_bucket_block() 