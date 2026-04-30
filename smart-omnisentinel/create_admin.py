import asyncio
import sys

sys.path.insert(0, '.')

async def create_admin():
    from core.security import hash_password
    from db.session import get_session_factory
    from db.models.user import User
    from sqlalchemy import select

    factory = get_session_factory()
    async with factory() as db:
        result = await db.execute(select(User).where(User.email == 'admin@demo.com'))
        if result.scalar_one_or_none():
            print('Admin already exists - OK')
            return

        user = User(
            email='admin@demo.com',
            hashed_password=hash_password('Admin123!'),
            full_name='Administrator',
            role='ADMIN',
            is_active=True,
        )
        db.add(user)
        await db.commit()

        print('Admin created successfully!')

asyncio.run(create_admin())