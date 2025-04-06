from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, DateTime, JSON
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    telegram_id = Column(Integer, primary_key=True)  # Исправлено: telegram_id как первичный ключ
    is_admin = Column(Boolean, default=False)
    conversations = relationship("Conversation", back_populates="user")

class Conversation(Base):
    __tablename__ = 'conversations'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.telegram_id'))  # Исправлена ссылка на внешний ключ
    start_time = Column(DateTime, default=datetime.now)
    end_time = Column(DateTime)
    is_successful = Column(Boolean)
    messages = Column(JSON, default=[])
    category = Column(String)
    user = relationship("User", back_populates="conversations")