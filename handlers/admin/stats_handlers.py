# app/handlers/admin/stats_handlers.py
from aiogram import Router, F
from aiogram.types import Message, BufferedInputFile, InputMediaPhoto
from aiogram.filters import Command, CommandObject
from app.database import sessions
from app.handlers.admin.stats import (
    generate_category_pie_chart,
    generate_total_conversations_plot,
    generate_requests_histogram
)
from app.database import crud
import logging

router = Router()


async def is_admin(message: Message) -> bool:
    async with sessions.AsyncSessionLocal() as db:
        user = await crud.get_or_create_user(db, message.from_user.id)
        return user.is_admin


@router.message(Command("stats_categories"))
async def categories_stats_handler(message: Message, command: CommandObject):
    if not await is_admin(message):
        return await message.answer("⛔ У вас нет прав доступа")

    # Парсим период из аргументов команды
    period = 7
    if command.args and command.args.isdigit():
        period = int(command.args)
        if period < 1 or period > 365:
            return await message.answer("❌ Период должен быть от 1 до 365 дней")

    async with sessions.AsyncSessionLocal() as db:
        try:
            # Генерируем график
            chart_buffer = await generate_category_pie_chart(db, period)

            # Создаем объект файла для отправки
            chart_file = BufferedInputFile(
                chart_buffer.getvalue(),
                filename=f"categories_{period}d.png"
            )

            # Отправляем результат
            await message.answer_photo(
                chart_file,
                caption=f"📊 Распределение диалогов по категориям за {period} дней"
            )
        except Exception as e:
            await message.answer(f"⚠️ Ошибка генерации отчета: {str(e)}")


@router.message(Command("stats_total"))
async def total_stats_handler(message: Message, command: CommandObject):
    if not await is_admin(message):
        return await message.answer("⛔ У вас нет прав доступа")

    period = 7
    if command.args and command.args.isdigit():
        period = int(command.args)
        if period < 1 or period > 365:
            return await message.answer("❌ Период должен быть от 1 до 365 дней")

    async with sessions.AsyncSessionLocal() as db:
        try:
            # Генерируем график
            chart_buffer = await generate_total_conversations_plot(db, period)

            chart_file = BufferedInputFile(
                chart_buffer.getvalue(),
                filename=f"total_{period}d.png"
            )

            await message.answer_photo(
                chart_file,
                caption=f"📈 Общее количество завершенных диалогов за {period} дней"
            )
        except Exception as e:
            await message.answer(f"⚠️ Ошибка генерации отчета: {str(e)}")


@router.message(Command("stats_full"))
async def full_stats_handler(message: Message):
    if not await is_admin(message):
        return await message.answer("⛔ У вас нет прав доступа")

    async with sessions.AsyncSessionLocal() as db:
        try:
            # Генерируем оба графика
            pie_buffer = await generate_category_pie_chart(db)
            total_buffer = await generate_total_conversations_plot(db)

            # Оборачиваем сгенерированные изображения в BufferedInputFile
            pie_file = BufferedInputFile(
                pie_buffer.getvalue(),
                filename="categories.png"
            )
            total_file = BufferedInputFile(
                total_buffer.getvalue(),
                filename="total.png"
            )

            media = [
                InputMediaPhoto(
                    media=pie_file,
                    caption="📊 Распределение по категориям"
                ),
                InputMediaPhoto(
                    media=total_file,
                    caption="📈 Общая статистика"
                )
            ]

            await message.answer_media_group(media)
        except Exception as e:
            await message.answer(f"⚠️ Ошибка генерации отчетов: {str(e)}")


@router.message(Command("stats_requests"))
async def requests_stats_handler(message: Message, command: CommandObject):
    if not await is_admin(message):
        return await message.answer("⛔ У вас нет прав доступа")

    period = 7
    if command.args and command.args.isdigit():
        period = int(command.args)
        if period < 1 or period > 60:
            return await message.answer("❌ Период должен быть от 1 до 60 дней")

    async with sessions.AsyncSessionLocal() as db:
        try:
            chart_buffer = await generate_requests_histogram(db, period)

            chart_file = BufferedInputFile(
                chart_buffer.getvalue(),
                filename=f"requests_{period}d.png"
            )

            await message.answer_photo(
                chart_file,
                caption=f"📅 Статистика запросов по дням за {period} дней"
            )
        except Exception as e:
            await message.answer(f"⚠️ Ошибка генерации отчета: {str(e)}")
            logging.error(f"Requests stats error: {str(e)}", exc_info=True)


@router.message(Command("help"))
async def help_handler(message: Message):
    if await is_admin(message):
        help_text = (
            "📊 Команды статистики для админов:\n"
            "/stats_categories [дней] - Распределение по категориям\n"
            "/stats_total [дней] - Общее количество диалогов\n"
            "/stats_requests [дней] - Запросы по дням\n"
            "/stats_full - Полный отчет\n"
            "/add_admin <id> - Назначить админа\n"
            "/remove_admin <id> - Удалить админа"
        )
        await message.answer(help_text)
