from __future__ import annotations

from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .config import settings
from sqlalchemy import text
from .models import Base


engine = create_async_engine(settings.DATABASE_URL, echo=False, pool_pre_ping=True, future=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


@asynccontextmanager
async def session_scope() -> AsyncSession:
    session = SessionLocal()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def init_db() -> None:
    async with engine.begin() as conn:
        # Ensure required extensions
        try:
            await conn.execute(text('CREATE EXTENSION IF NOT EXISTS citext'))
        except Exception:
            pass
        await conn.run_sync(Base.metadata.create_all)
