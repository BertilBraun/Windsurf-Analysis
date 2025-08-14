from __future__ import annotations

import boto3
from botocore.client import Config

from server.backend.config import Settings


def s3_client():
    session = boto3.Session()
    client = session.client(
        's3',
        aws_access_key_id=Settings.S3_ACCESS_KEY_ID,
        aws_secret_access_key=Settings.S3_SECRET_ACCESS_KEY,
        endpoint_url=Settings.S3_ENDPOINT_URL,
        region_name=Settings.S3_REGION,
        config=Config(s3={'addressing_style': 'virtual'}),
    )
    return client


def object_url(key: str) -> str:
    # s3://bucket/key style URL stored in DB
    return f's3://{Settings.S3_BUCKET}/{key}'
