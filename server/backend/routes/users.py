from __future__ import annotations
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from passlib.context import CryptContext

from server.backend.config import Settings
from server.backend.models import User
from server.backend.database.db import get_db
from server.backend.database.accessor import DatabaseAccessor


router = APIRouter(prefix='/admin', tags=['users'])


class CreateUserRequest(BaseModel):
    secret: str
    email: str
    password: str


class CreateUserResponse(BaseModel):
    status: Literal['success', 'error']
    id: str
    email: str


@router.post('/users', response_model=CreateUserResponse)
async def create_user(payload: CreateUserRequest, db: DatabaseAccessor = Depends(get_db)):
    if payload.secret != Settings.USER_CREATE_SECRET:
        raise HTTPException(status_code=401, detail='invalid secret')

    pwd = CryptContext(schemes=['bcrypt'], deprecated='auto')
    existing = await db.get_user_by_email(payload.email)
    if existing is not None:
        raise HTTPException(status_code=409, detail='email already exists')

    user = User(email=payload.email, password_hash=pwd.hash(payload.password))
    await db.add(user)

    return CreateUserResponse(status='success', id=str(user.id), email=user.email)
