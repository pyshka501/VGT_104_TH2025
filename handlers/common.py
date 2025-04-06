from aiogram.types import Message
from app.database import crud, sessions


async def is_admin(message: Message) -> bool:
    async with sessions.AsyncSessionLocal() as db:
        user = await crud.get_or_create_user(db, message.from_user.id)
        return user.is_admin
