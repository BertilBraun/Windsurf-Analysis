"""
Feedback routes for handling user feedback submissions.
"""
from __future__ import annotations


from fastapi import APIRouter, Depends
from pydantic import BaseModel

from auth.firebase_auth import User, get_current_user
from repos.reports_repo import ReportsRepo


router = APIRouter(prefix='/feedback', tags=['feedback'])
reports_repo = ReportsRepo()


class FeedbackCreateRequest(BaseModel):
    """
    Request schema for submitting user feedback.
    """
    message: str


@router.post('')
def create_feedback(payload: FeedbackCreateRequest, user: User = Depends(get_current_user)):
    """
    Creates a feedback report for the current authenticated user.

    Args:
        payload: The feedback submission payload containing the message.
        user: The authenticated user submitting the feedback.

    Returns:
        A dictionary indicating the success status of the operation.
    """
    reports_repo.add_report(user.uid, 'feedback_id', 'feedback', payload.message)
    return {'ok': True}
