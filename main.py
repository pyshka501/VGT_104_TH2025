# main.py
import asyncio
from aiogram import Bot, Dispatcher
from app.handlers.user.init import router as init_router_user
from aiogram.fsm.storage.memory import MemoryStorage
from app.handlers.admin.init import router as init_router_admin
from app.handlers.admin.stats_handlers import router as stats_router_admin
import logging
from app.database.sessions import engine, AsyncSessionLocal
from app.database import models
from config import TOKEN



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)


async def init_db():
    """Инициализация базы данных"""
    async with engine.begin() as conn:
        await conn.run_sync(models.Base.metadata.create_all)
    logging.info("Database initialized")

from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
async def main():
    """Основная функция запуска бота"""
    await init_db()

    bot = Bot(token=TOKEN)#, default=DefaultBotProperties(parse_mode=ParseMode.HTML)
    storage = MemoryStorage()

    dp = Dispatcher(storage=storage, bot=bot)
    dp.include_router(init_router_user)
    dp.include_router(init_router_admin)
    dp.include_router(stats_router_admin)
    
    await dp.start_polling(bot)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info('Ctrl+C pressed. Stopping.')
