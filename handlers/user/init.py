# app/handlers/user/init.py
from aiogram import Router, F
from aiogram.types import Message, CallbackQuery
from aiogram.filters import Command
from app.handlers.common import is_admin
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload  # импортируем опцию для жадной загрузки
from app.database import crud, sessions, models
from app.keyboards import keyboards
import logging

from app.LLM_folder.rag_model import rag

# rag = RAGSystem("config.yaml")
rag.load_data()
router = Router()


# Определяем два состояния для FSM:
# awaiting_initial – ожидание первичной обратной связи (лайк/дизлайк)
# awaiting_secondary – ожидание вторичной обратной связи (request_human/reask)
class FeedbackState(StatesGroup):
    awaiting_initial = State()
    awaiting_secondary = State()


@router.message(Command("start"))
async def start_handler(message: Message, state: FSMContext):
    await state.clear()
    async with sessions.AsyncSessionLocal() as db:
        await crud.get_or_create_user(db, message.from_user.id)
        conversation = await crud.get_active_conversation(db, message.from_user.id)
        if conversation:
            await crud.complete_conversation(db, conversation, False)

    await message.answer("👋 Привет! Я AI-ассистент. Задай мне вопрос, и я постараюсь помочь!")


@router.message(~F.text.startswith('/') & F.text)
async def text_message_handler(message: Message, state: FSMContext):
    if await is_admin(message):
        return
    # a = Message("Мы уже бежим Вам на помощь!")
    # await message.bot.send_message(message.chat_id, a)
    current_state = await state.get_state()
    if current_state:
        # Пользователь находится в одном из состояний ожидания – не обрабатываем новое сообщение
        await message.answer(
            "Пожалуйста, завершите обратную связь (нажмите соответствующую кнопку), прежде чем продолжить диалог."
        )
        return
        
    bot_response, theme = rag.query(message.text)
    # await message.bot.delete_message(message.chat_id, a)
    
    async with sessions.AsyncSessionLocal() as db:
        user = await crud.get_or_create_user(db, message.from_user.id)
        conversation = await crud.get_active_conversation(db, user.telegram_id)

        if not conversation:
            conversation = await crud.create_conversation(db, user.telegram_id)
            conversation = await crud.add_category_to_conversation(db, conversation, theme)

        conversation = await crud.add_message_to_conversation(db, conversation, message.text, False)
        
        # response, theme = rag.query(query)
         # ЗАМЕНИТЬ
        # conversation = await crud.add_category_to_conversation(db, conversation, message.text)
        
        conversation = await crud.add_message_to_conversation(db, conversation, bot_response.replace("<p>", "").replace("</p>", ""), True)

        await state.set_state(FeedbackState.awaiting_initial)

        await message.answer(bot_response.replace("<p>", "").replace("</p>", ""), reply_markup=keyboards.get_feedback_kb())


@router.callback_query(F.data == "like")
async def like_handler(callback: CallbackQuery, state: FSMContext):

    current_state = await state.get_state()
    if current_state != FeedbackState.awaiting_initial.state:
        await callback.answer("Действие недопустимо в текущем состоянии.", show_alert=True)
        return

    async with sessions.AsyncSessionLocal() as db:
        user = await crud.get_or_create_user(db, callback.from_user.id)
        conversation = await crud.get_active_conversation(db, user.telegram_id)
        if conversation:
            await crud.complete_conversation(db, conversation, True)

    await state.clear()

    await callback.message.edit_reply_markup()
    await callback.answer()
    await callback.message.answer("✅ Рад, что смог помочь! Обращайтесь еще!")


@router.callback_query(F.data == "dislike")
async def dislike_handler(callback: CallbackQuery, state: FSMContext):

    current_state = await state.get_state()
    if current_state != FeedbackState.awaiting_initial.state:
        await callback.answer("Действие недопустимо в текущем состоянии.", show_alert=True)
        return

    async with sessions.AsyncSessionLocal() as db:
        user = await crud.get_or_create_user(db, callback.from_user.id)
        conversation = await crud.get_active_conversation(db, user.telegram_id)

        # Убрано завершение диалога здесь
        # if conversation:
        #    await crud.complete_conversation(db, conversation, False)
    await state.set_state(FeedbackState.awaiting_secondary)
    await callback.message.edit_reply_markup()
    await callback.answer()
    await callback.message.answer(
        "❌ Извините, что ответ не подошел. Выберите действие:",
        reply_markup=keyboards.get_feedback_options_kb()
    )


@router.callback_query(F.data == "request_human")
async def human_handler(callback: CallbackQuery, state: FSMContext):

    current_state = await state.get_state()
    if current_state != FeedbackState.awaiting_secondary.state:
        await callback.answer("Действие недопустимо в текущем состоянии.", show_alert=True)
        return


    async with sessions.AsyncSessionLocal() as db:
        user = await crud.get_or_create_user(db, callback.from_user.id)
        conversation = await crud.get_active_conversation(db, user.telegram_id)

        if conversation:
            # Завершаем диалог только при обращении к консультанту
            await crud.complete_conversation(db, conversation, False)
            await notify_admins(callback.bot, conversation)

    await state.clear()
    await callback.message.edit_reply_markup()
    await callback.answer()
    await callback.message.answer("🆘 Ваш запрос передан консультанту. Ожидайте ответа.")


@router.callback_query(F.data == "reask")
async def reask_handler(callback: CallbackQuery, state: FSMContext):

    current_state = await state.get_state()
    if current_state != FeedbackState.awaiting_secondary.state:
        await callback.answer("Действие недопустимо в текущем состоянии.", show_alert=True)
        return

    await state.clear()
    await callback.message.delete()
    await callback.message.answer("🔄 Пожалуйста, задайте вопрос еще раз:")
    await callback.answer()


async def notify_admins(bot, conversation):
    try:
        async with sessions.AsyncSessionLocal() as db:
            # Получаем диалог заново в новой сессии и загружаем связь user
            result = await db.execute(
                select(models.Conversation)
                .options(selectinload(models.Conversation.user))
                .where(models.Conversation.id == conversation.id)
            )
            conversation = result.scalars().first()

            if not conversation:
                logging.warning("Conversation not found")
                return

            result = await db.execute(
                select(models.User).where(models.User.is_admin == True)
            )
            admins = result.scalars().all()

            if not admins:
                logging.warning("No admins found for notification")
                return

            history = "\n".join(
                [f"{'Bot' if m['is_bot'] else 'User'}: {m['text']}"
                 for m in conversation.messages]
            )

            for admin in admins:
                try:
                    await bot.send_message(
                        chat_id=admin.telegram_id,
                        text=(
                            f"🚨 НОВЫЙ ЗАПРОС ПОДДЕРЖКИ\n"
                            f"User ID: {conversation.user.telegram_id}\n"
                            f"История диалога:\n{history}"
                        )
                    )
                    logging.info(f"Notification sent to admin {admin.telegram_id}")
                except Exception as e:
                    logging.error(f"Error sending to admin {admin.telegram_id}: {str(e)}")

    except Exception as e:
        logging.error(f"Error in notify_admins: {str(e)}", exc_info=True)
