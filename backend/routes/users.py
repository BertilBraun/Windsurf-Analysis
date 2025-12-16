from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from auth.firebase_auth import User, get_current_user
from repos.user_repo import UserRepo

router = APIRouter(prefix='/users', tags=['users'])
user_repo = UserRepo()


@router.post('/{user_id}')
def create_user(user_id: str, user: User = Depends(get_current_user)):
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
