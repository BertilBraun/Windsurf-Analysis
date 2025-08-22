from __future__ import annotations

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from sqlalchemy import text

from server.backend.models import Base
from server.backend.config import Settings
from server.backend.database.accessor import DatabaseAccessor

engine = create_async_engine(
    Settings.DATABASE_URL,  # + '?sslmode=require', # Does not seem to work with Neon
    echo=False,
    # keep a small reusable pool; don’t reconnect every request
    pool_size=5,
    pool_pre_ping=True,
    connect_args={'statement_cache_size': 0},  # disable asyncpg’s prepared statement cache
    future=True,
)
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
            if session.in_transaction():
                transaction = session.get_transaction()
                if transaction and transaction.is_active:
                    # Only commit if you actually did DML; for SELECT-only, nothing to do
                    await session.commit()
        except Exception:
            if session.in_transaction():
                await session.rollback()
            raise
