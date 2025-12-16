from __future__ import annotations

from google.cloud import firestore

from db.firestore_client import reports


class ReportsRepo:
    def add_report(self, user_id: str, job_id: str, report_type: str, message: str) -> None:
        reports.add(
            {
                'user_id': user_id,
                'job_id': job_id,
                'type': report_type,
                'message': message,
                'created_at': firestore.SERVER_TIMESTAMP,
            },
        )
