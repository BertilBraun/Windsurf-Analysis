from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from auth.firebase_auth import User, get_current_user, get_current_user_without_email_verification
from firebase_admin import auth as firebase_admin_auth
from db.firestore_client import now
from repos.user_repo import UserRepo
from repos.user_jobs_repo import UserJobsRepo

router = APIRouter(prefix='/users', tags=['users'])
user_repo = UserRepo()
user_jobs_repo = UserJobsRepo()


class CreateUserRequest(BaseModel):
    terms_accepted: bool | None = None
    marketing_consent: bool | None = None


class UpdateConsentRequest(BaseModel):
    terms_accepted: bool | None = None
    marketing_consent: bool | None = None


@router.post('/{user_id}')
def create_user(
    user_id: str,
    payload: CreateUserRequest | None = None,
    user: User = Depends(get_current_user_without_email_verification),
):
    if user.uid != user_id:
        raise HTTPException(status_code=403, detail='Forbidden')
    if user_repo.does_user_exist(user_id):
        return {'ok': True, 'detail': 'User already exists'}
    if payload and payload.terms_accepted is False:
        raise HTTPException(status_code=400, detail='Terms must be accepted')

    terms_accepted_at = now() if payload and payload.terms_accepted else None
    privacy_accepted_at = terms_accepted_at
    marketing_consent = payload.marketing_consent if payload else None
    marketing_consent_at = now() if payload and payload.marketing_consent else None

    user_repo.create_user(
        user_id,
        user.email,
        terms_accepted_at=terms_accepted_at,
        privacy_accepted_at=privacy_accepted_at,
        marketing_consent=marketing_consent,
        marketing_consent_at=marketing_consent_at,
    )
    return {'ok': True}


@router.patch('/{user_id}/consent')
def update_user_consent(
    user_id: str,
    payload: UpdateConsentRequest,
    user: User = Depends(get_current_user_without_email_verification),
):
    if user.uid != user_id:
        raise HTTPException(status_code=403, detail='Forbidden')
    if payload.terms_accepted is False:
        raise HTTPException(status_code=400, detail='Terms must be accepted')

    if not user_repo.does_user_exist(user_id):
        terms_accepted_at = None
        privacy_accepted_at = None
        if payload.terms_accepted:
            ts = now()
            terms_accepted_at = ts
            privacy_accepted_at = ts
        marketing_consent = payload.marketing_consent if payload.marketing_consent is not None else None
        marketing_consent_at = now() if payload.marketing_consent else None

        user_repo.create_user(
            user_id,
            user.email,
            terms_accepted_at=terms_accepted_at,
            privacy_accepted_at=privacy_accepted_at,
            marketing_consent=marketing_consent,
            marketing_consent_at=marketing_consent_at,
        )
        return {'ok': True, 'created': True}

    fields: dict = {}
    if payload.terms_accepted:
        ts = now()
        fields['terms_accepted_at'] = ts
        fields['privacy_accepted_at'] = ts
    if payload.marketing_consent is not None:
        fields['marketing_consent'] = payload.marketing_consent
        fields['marketing_consent_at'] = now() if payload.marketing_consent else None

    user_repo.update_user_fields(user_id, fields)
    return {'ok': True}


@router.get('/{user_id}')
def get_user(user_id: str, user: User = Depends(get_current_user)):
    if user.uid != user_id:
        raise HTTPException(status_code=403, detail='Forbidden')
    return user_repo.get_user(user_id)


@router.delete('/{user_id}')
def delete_user(user_id: str, user: User = Depends(get_current_user)):
    if user.uid != user_id:
        raise HTTPException(status_code=403, detail='Forbidden')

    # 1) Delete user docs first (keeps cleanup even if auth deletion fails)
    deleted_user_jobs = user_jobs_repo.delete_all_for_user(user_id)
    user_repo.delete_user(user_id)

    # 2) Delete Firebase Auth user (revokes ability to sign in)
    try:
        firebase_admin_auth.delete_user(user_id)
    except firebase_admin_auth.UserNotFoundError:
        # Treat as already deleted.
        pass

    return {'ok': True, 'deleted_user_jobs': deleted_user_jobs}
