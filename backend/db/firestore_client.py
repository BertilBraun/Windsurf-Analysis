from __future__ import annotations

import datetime

from google.cloud import firestore

from config import settings


db: firestore.Client = firestore.Client(database=settings.firestore_database)

jobs = db.collection('jobs')
users = db.collection('users')
user_jobs = db.collection('user_jobs')
reports = db.collection('reports')

def now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)
