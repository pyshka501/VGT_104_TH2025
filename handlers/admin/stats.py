# app/handlers/admin/stats.py
from sqlalchemy.future import select
from sqlalchemy import func, and_
from datetime import datetime, timedelta
from io import BytesIO
from app.database import models
import matplotlib.pyplot as plt


async def get_conversation_stats(db, period: int = 7):
    """Получение статистики по категориям за указанный период"""
    now = datetime.now()
    start_date = now - timedelta(days=period)
    stmt = (
        select(
            models.Conversation.category,
            func.count(models.Conversation.id).label('count')
        )
        .where(
            and_(
                models.Conversation.start_time >= start_date,
                models.Conversation.end_time.is_not(None)
            )
        )
        .group_by(models.Conversation.category)
    )
    result = await db.execute(stmt)
    stats = result.all()
    return {category: count for category, count in stats}


async def generate_category_pie_chart(db, period: int = 7):
    """Генерация круговой диаграммы по категориям"""
    stats = await get_conversation_stats(db, period)
    categories = list(stats.keys())
    counts = list(stats.values())

    plt.figure(figsize=(10, 8))
    plt.pie(
        counts,
        labels=categories,
        autopct='%1.1f%%',
        startangle=90,
        colors=plt.cm.Paired.colors,
        wedgeprops={'edgecolor': 'white', 'linewidth': 0.5}
    )
    plt.title(f'Распределение диалогов по категориям\n(Период: {period} дней)', pad=20)

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf


async def get_total_conversations(db, period: int = 7) -> int:
    """Получение общего количества диалогов за период"""
    now = datetime.now()
    start_date = now - timedelta(days=period)
    stmt = (
        select(func.count(models.Conversation.id))
        .where(
            and_(
                models.Conversation.start_time >= start_date,
                models.Conversation.end_time.is_not(None)
            )
        )
    )
    result = await db.execute(stmt)
    return result.scalar()


async def generate_total_conversations_plot(db, period: int = 7) -> BytesIO:
    """Генерация графика с общим количеством диалогов"""
    total = await get_total_conversations(db, period)

    # Используем корректный стиль оформления
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.text(
        0.5, 0.6,
        f'{total}',
        fontsize=48,
        ha='center',
        va='center',
        color='#2c7bb6'
    )
    ax.text(
        0.5, 0.3,
        f'диалогов за {period} дней',
        fontsize=14,
        ha='center',
        va='center',
        color='#4a4a4a'
    )
    ax.axis('off')

    buf = BytesIO()
    plt.savefig(
        buf,
        format='png',
        dpi=100,
        bbox_inches='tight',
        pad_inches=0.2
    )
    buf.seek(0)
    plt.close(fig)
    return buf


async def generate_requests_histogram(db, period: int = 7) -> BytesIO:
    """Генерация гистограммы запросов по дням"""
    # Получаем данные из БД
    now = datetime.now().date()
    dates = [now - timedelta(days=i) for i in range(period - 1, -1, -1)]

    stmt = (
        select(
            func.date(models.Conversation.start_time).label('day'),
            func.count(models.Conversation.id).label('count')
        )
        .where(
            models.Conversation.start_time >= now - timedelta(days=period),
            models.Conversation.end_time.is_not(None)
        )
        .group_by(func.date(models.Conversation.start_time)))

    result = await db.execute(stmt)
    db_data = {row.day: row.count for row in result.all()}

    # Формируем данные для графика
    counts = [db_data.get(date.strftime('%Y-%m-%d'), 0) for date in dates]
    labels = [date.strftime('%d.%m') for date in dates]

    # Создаем график
    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(
        labels,
        counts,
        color='#4c72b0',
        edgecolor='white',
        linewidth=1.5
    )

    # Добавляем значения поверх столбцов
    for bar in bars:
        height = bar.get_height()
    ax.annotate(f'{height}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center',
                va='bottom',
                fontsize=10
                )

    # Настраиваем внешний вид
    ax.set_title(f'Количество запросов по дням (последние {period} дней)', pad=20, fontsize=14)
    ax.set_xlabel('Дата', labelpad=15, fontsize=12)
    ax.set_ylabel('Количество запросов', labelpad=15, fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Сохраняем в буфер
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=120)
    buf.seek(0)
    plt.close(fig)

    return buf
