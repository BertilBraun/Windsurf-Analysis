import asyncio
import os
import sys

from getpass import getpass
from passlib.context import CryptContext

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.db import SessionLocal, init_db  # noqa: E402
from app.models import User  # noqa: E402


pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')


async def main():
    await init_db()
    username = input('Username: ')
    password = getpass('Password: ')
    password2 = getpass('Confirm: ')
    if password != password2:
        print('Passwords do not match', file=sys.stderr)
        sys.exit(1)

    password_hash = pwd_context.hash(password)

    async with SessionLocal() as session:
        user = User(username=username, password_hash=password_hash)
        session.add(user)
        await session.commit()
    print('Created user', username)


if __name__ == '__main__':
    asyncio.run(main())
