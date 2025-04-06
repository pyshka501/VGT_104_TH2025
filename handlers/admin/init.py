from aiogram import Router
from aiogram.types import Message
from aiogram.filters import Command, CommandObject
from app.database import crud, sessions, models
from app.handlers.common import is_admin
from sqlalchemy.future import select
import logging

logger = logging.getLogger(__name__)
router = Router()


@router.message(Command("send"))
async def send_message_to_user(message: Message, command: CommandObject):
    """
    Handler для отправки сообщения пользователю по его ID.
    Команда имеет формат: /send <user_id> <сообщение>
    Только администраторы могут использовать эту команду.
    """
    # Проверяем, что отправитель является администратором
    if not await is_admin(message):
        return

    # Проверяем наличие аргументов и разделяем их на user_id и текст сообщения
    if not command.args:
        return await message.answer("❌ Укажите ID пользователя и сообщение: /send <user_id> <сообщение>")

    args = command.args.split(maxsplit=1)
    if len(args) < 2:
        return await message.answer("❌ Неверный формат. Используйте: /send <user_id> <сообщение>")

    user_id_str, user_message = args
    if not user_id_str.isdigit():
        return await message.answer("❌ ID пользователя должен быть числом.")

    user_message = "Ответ от оператора:\n" + user_message

    target_id = int(user_id_str)

    try:
        # Отправляем сообщение пользователю с заданным ID
        await message.bot.send_message(target_id, user_message)
        await message.answer(f"✅ Сообщение отправлено пользователю {target_id}.")
    except Exception as e:
        logger.error(f"Ошибка при отправке сообщения пользователю {target_id}: {str(e)}")
        await message.answer(f"❌ Не удалось отправить сообщение пользователю {target_id}.")


@router.message(Command("report"))
async def report_handler(message: Message):
    async with sessions.AsyncSessionLocal() as db:
        user = await crud.get_or_create_user(db, message.from_user.id)
        if not user.is_admin:
            return

        result = await db.execute(
            select(models.Conversation).filter(
                models.Conversation.is_successful == False
            )
        )
        conversations = result.scalars().all()

        if not conversations:
            await message.answer("⛔ Нет активных запросов в поддержку")
            return

        for conv in conversations:
            await db.refresh(conv, ['user'])

            history = "\n".join([f"{'Bot' if m['is_bot'] else 'User'}: {m['text']}" for m in conv.messages])
            await message.answer(
                f"🆘 Запрос в поддержку #ID{conv.id}\n"
                f"User ID: {conv.user.telegram_id}\n"
                f"История:\n{history}"
            )


@router.message(Command("add_admin"))
async def add_admin_handler(message: Message, command: CommandObject):
    if await is_admin(message):
        return

    if not command.args or not command.args.isdigit():
        return await message.answer("❌ Укажите ID пользователя: /add_admin <user_id>")

    target_id = int(command.args)
    async with sessions.AsyncSessionLocal() as db:
        target_user = await crud.update_user_admin_status(db, target_id, True)

    if target_user:
        await message.answer(f"✅ Пользователь {target_id} назначен администратором")
        await message.bot.send_message(
            target_id,
            "🎉 Вам выданы права администратора!"
        )
    else:
        await message.answer("❌ Пользователь не найден")


@router.message(Command("remove_admin"))
async def remove_admin_handler(message: Message, command: CommandObject):
    if not await is_admin(message):
        return

    if not command.args or not command.args.isdigit():
        return await message.answer("❌ Укажите ID пользователя: /remove_admin <user_id>")

    target_id = int(command.args)
    if message.from_user.id == target_id:
        return await message.answer("❌ Нельзя снять права с самого себя")

    async with sessions.AsyncSessionLocal() as db:
        target_user = await crud.update_user_admin_status(db, target_id, False)

    if target_user:
        await message.answer(f"✅ Пользователь {target_id} лишен прав администратора")
        await message.bot.send_message(
            target_id,
            "😞 Ваши права администратора были отозваны"
        )
    else:
        await message.answer("❌ Пользователь не найден")
