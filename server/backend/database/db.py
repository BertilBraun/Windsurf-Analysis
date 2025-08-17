from __future__ import annotations

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from sqlalchemy import text

from server.backend.models import Base
from server.backend.config import Settings
from server.backend.database.accessor import DatabaseAccessor

engine = create_async_engine(Settings.DATABASE_URL, echo=False, pool_pre_ping=True, future=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def init_db() -> None:
    async with engine.begin() as conn:
        # Ensure required extensions
        try:
            await conn.execute(text('CREATE EXTENSION IF NOT EXISTS citext'))
        except Exception:
            pass
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncGenerator[DatabaseAccessor, None]:
    async with SessionLocal() as session:
        try:
            yield DatabaseAccessor(session)
            await session.commit()  # Commit successful transactions
        except Exception:
            await session.rollback()  # Rollback on error
            raise
