from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from auth.firebase_auth import User, get_current_user, get_current_user_without_email_verification
from firebase_admin import auth as firebase_admin_auth
from repos.user_repo import UserRepo
from repos.user_jobs_repo import UserJobsRepo

router = APIRouter(prefix='/users', tags=['users'])
user_repo = UserRepo()
user_jobs_repo = UserJobsRepo()


@router.post('/{user_id}')
def create_user(user_id: str, user: User = Depends(get_current_user_without_email_verification)):
    if user.uid != user_id:
        raise HTTPException(status_code=403, detail='Forbidden')
    if user_repo.does_user_exist(user_id):
        raise HTTPException(status_code=400, detail='User already exists')
    user_repo.create_user(user_id, user.email)
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
