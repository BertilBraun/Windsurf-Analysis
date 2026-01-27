"""Firestore database client and collection references."""
from __future__ import annotations

import datetime

from google.cloud import firestore

from config import settings


# Initialize Firestore client
db: firestore.Client = firestore.Client(database=settings.firestore_database)

# Collection references
jobs = db.collection('jobs')
users = db.collection('users')
user_jobs = db.collection('user_jobs')
reports = db.collection('reports')

def now() -> datetime.datetime:
    """Returns the current UTC timestamp."""
    return datetime.datetime.now(datetime.timezone.utc)
