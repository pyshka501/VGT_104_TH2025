# app/database/crud.py
from sqlalchemy.future import select
from datetime import datetime
from app.database import models


async def get_or_create_user(db, telegram_id: int):
    result = await db.execute(
        select(models.User).where(models.User.telegram_id == telegram_id)
    )
    user = result.scalars().first()

    if not user:
        user = models.User(telegram_id=telegram_id)
        db.add(user)
        await db.commit()
        await db.refresh(user)
    return user


async def create_conversation(db, telegram_id: int):
    conversation = models.Conversation(user_id=telegram_id)
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)
    return conversation


async def get_active_conversation(db, telegram_id: int):
    result = await db.execute(
        select(models.Conversation)
        .where(
            models.Conversation.user_id == telegram_id,
            models.Conversation.end_time == None
        )
    )
    return result.scalars().first()


async def add_message_to_conversation(db, conversation: models.Conversation, text: str, is_bot: bool):
    new_message = {
        "text": text,
        "is_bot": is_bot,
        "timestamp": datetime.now().isoformat()
    }
    conversation.messages = [*conversation.messages, new_message]
    await db.commit()
    await db.refresh(conversation)
    return conversation


async def complete_conversation(db, conversation: models.Conversation, is_successful: bool):
    conversation.end_time = datetime.now()
    conversation.is_successful = is_successful
    await db.commit()
    await db.refresh(conversation)
    return conversation


async def update_user_admin_status(db, telegram_id: int, is_admin: bool):
    result = await db.execute(
        select(models.User).where(models.User.telegram_id == telegram_id)
    )
    user = result.scalars().first()

    if user:
        user.is_admin = is_admin
        await db.commit()
        await db.refresh(user)
    return user


async def add_category_to_conversation(db, conversation: models.Conversation, message_text: str):
    if not conversation.category:
        conversation.category = message_text # заменить на нейронку!
        await db.commit()
        await db.refresh(conversation)
    return conversation

def get_conversation_duration(conversation: models.Conversation):
    if conversation.end_time:
        return (conversation.end_time - conversation.start_time).total_seconds()
    return None
