from __future__ import annotations
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from passlib.context import CryptContext
from sqlalchemy import select

from server.backend.config import Settings
from server.backend.db import session_scope
from server.backend.models import User


router = APIRouter(prefix='/admin', tags=['users'])


class CreateUserRequest(BaseModel):
    secret: str
    email: str
    password: str


class CreateUserResponse(BaseModel):
    status: Literal['success', 'error']
    email: str


@router.post('/users', response_model=CreateUserResponse)
async def create_user(payload: CreateUserRequest):
    if payload.secret != Settings.USER_CREATE_SECRET:
        raise HTTPException(status_code=401, detail='invalid secret')

    pwd = CryptContext(schemes=['bcrypt'], deprecated='auto')
    async with session_scope() as db:
        existing = (await db.execute(select(User).where(User.email == payload.email))).scalars().first()
        if existing:
            raise HTTPException(status_code=409, detail='email already exists')

        user = User(email=payload.email, password_hash=pwd.hash(payload.password))
        db.add(user)
        return CreateUserResponse(status='success', email=user.email)
