from __future__ import annotations
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from server.backend.config import Settings
from server.backend.models import User
from server.backend.db import get_db


router = APIRouter(prefix='/admin', tags=['users'])


class CreateUserRequest(BaseModel):
    secret: str
    email: str
    password: str


class CreateUserResponse(BaseModel):
    status: Literal['success', 'error']
    id: str
    email: str


async def _get_user_by_email(db: AsyncSession, email: str) -> User | None:
    res = await db.execute(select(User).where(User.email == email))
    return res.scalar_one_or_none()


@router.post('/users', response_model=CreateUserResponse)
async def create_user(payload: CreateUserRequest, db: AsyncSession = Depends(get_db)):
    if payload.secret != Settings.USER_CREATE_SECRET:
        raise HTTPException(status_code=401, detail='invalid secret')

    pwd = CryptContext(schemes=['bcrypt'], deprecated='auto')
    existing = await _get_user_by_email(db, payload.email)
    if existing is not None:
        raise HTTPException(status_code=409, detail='email already exists')

    user = User(email=payload.email, password_hash=pwd.hash(payload.password))
    db.add(user)
    await db.flush()  # Flush to get the user id

    return CreateUserResponse(status='success', id=str(user.id), email=user.email)
