from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.utils.keyboard import InlineKeyboardBuilder


def get_feedback_kb():
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="👍", callback_data="like"),
            InlineKeyboardButton(text="👎", callback_data="dislike")
        ]
    ])

def get_feedback_options_kb():
    builder = InlineKeyboardBuilder()
    builder.button(text="🔄 Переспросить", callback_data="reask")
    builder.button(text="👨💻 Консультант", callback_data="request_human")
    builder.adjust(1)
    return builder.as_markup()