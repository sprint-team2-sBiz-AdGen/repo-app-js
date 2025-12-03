
"""데이터베이스 모델 및 세션 관리"""

import datetime
from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from sqlalchemy.dialects.postgresql import UUID, JSONB
from config import DATABASE_URL

Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class TestAsset(Base):
    """테스트용 Asset 테이블 (간단한 insert/delete 테스트용)"""
    __tablename__ = "test_assets"
    
    asset_id = Column(UUID(as_uuid=True), primary_key=True)
    tenant_id = Column(String(255), nullable=False)
    asset_url = Column(String(500), nullable=False)
    asset_kind = Column(String(100))  # forbidden_mask, final, etc.
    width = Column(Integer)
    height = Column(Integer)
    created_at = Column(DateTime(timezone=True), default=datetime.datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)


def get_db():
    """DB 세션 의존성"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

