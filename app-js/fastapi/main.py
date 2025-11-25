import os
import json
import uuid
from fastapi import UploadFile, File
from contextlib import asynccontextmanager
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
    create_engine,
)
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session

from openai import OpenAI

# ============================
# 🔐 Environment & Config
# ============================

load_dotenv()

POSTGRES_USER = os.getenv("POSTGRES_USER", "feedly_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "feedly_password")
POSTGRES_DB = os.getenv("POSTGRES_DB", "feedly_db")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres-js")  # docker service name
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5436")         # your custom port

DATABASE_URL = (
    f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

client = OpenAI(api_key=OPENAI_API_KEY)

MEDIA_ROOT = os.getenv("MEDIA_ROOT", "/app/media/uploads")
os.makedirs(MEDIA_ROOT, exist_ok=True)

# ============================
# 💾 SQLAlchemy Setup
# ============================

Base = declarative_base()
engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class UserDescription(Base):
    __tablename__ = "user_descriptions"

    id = Column(Integer, primary_key=True, index=True)
    description_kr = Column(Text, nullable=False)
    description_en = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    ad_copies = relationship("AdCopy", back_populates="description")

class UploadedImage(Base):
    __tablename__ = "uploaded_images"

    id = Column(Integer, primary_key=True, index=True)
    file_path = Column(String(512), nullable=False)
    original_filename = Column(String(255), nullable=False)
    content_type = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    ad_copies = relationship("AdCopy", back_populates="image")

class AdCopy(Base):
    __tablename__ = "ad_copies"

    id = Column(Integer, primary_key=True, index=True)
    description_id = Column(Integer, ForeignKey("user_descriptions.id"), nullable=False)
    strategy_id = Column(Integer, nullable=False)
    strategy_name = Column(String(100), nullable=False)
    product_name = Column(Text, nullable=False)
    copy_ko = Column(Text, nullable=False)
    image_id = Column(Integer, ForeignKey("uploaded_images.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    description = relationship("UserDescription", back_populates="ad_copies")
    image = relationship("UploadedImage", back_populates="ad_copies")

def init_db():
    """데이터베이스 테이블을 생성합니다."""
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully.")
    except Exception as e:
        print(f"❌ Error creating database tables: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 앱의 시작과 종료 이벤트를 관리합니다."""
    print("🚀 Application startup...")
    init_db()
    yield
    print("👋 Application shutdown.")

# ============================
# 🧠 GPT System Prompts
# ============================

TRANSLATION_SYSTEM_PROMPT = """
You are a professional marketing copy translator.

Task:
- Input: a Korean sentence or short paragraph written by a small F&B business owner
  describing their menu item, store concept, or selling point.
- Output: a NATURAL English version that sounds like it was originally written in English
  for an Instagram ad context.

Rules:
- Keep meaning and nuance, but do NOT translate word-by-word.
- Make it sound smooth and marketing-friendly.
- One or two sentences maximum.
- Output JSON ONLY:
  { "translation_en": "..." }
"""

GPT_COPY_SYSTEM_PROMPT = """
You are Feedly AI, an AI assistant that generates Korean Instagram ad copy
for small F&B business owners.

You will be given:
- strategy_id (1–8)
- strategy_name (in English)
- product_name (in Korean or English)
- user_description_kr: Korean text from the owner
- user_description_en: English translation of that text
- foreground_analysis: a description of the masked food subject from LLAVA (may be empty).

Your tasks:
1. Generate THREE different Korean ad copy options for Instagram.
2. Each option must contain:
   - "copy_ko": a natural, marketing-friendly Korean sentence or two.

Strategy tones:
  1. Hero Dish Focus — emphasize visual deliciousness & texture.
  2. Seasonal / Limited — urgency + seasonal mood.
  3. Behind-the-Scenes — sincerity, craftsmanship.
  4. Lifestyle — cozy, everyday scene.
  5. UGC / Social Proof — authentic, casual customer vibe.
  6. Minimalist Branding — clean, premium, fewer words.
  7. Emotion / Comfort — warm, nostalgic.
  8. Retro / Vintage — storytelling, old-days atmosphere.

Rules:
- Output ONLY Korean copy for each variant.
- For Minimalist Branding (strategy_id 6): use exactly ONE short sentence, clean and restrained.
- For other strategies: 1–2 sentences per variant.
- Each of the 3 variants must feel clearly different (focus, angle, nuance).
- Do NOT include hashtags or emojis.
- Output JSON ONLY, with this format:

{
  "variants": [
    { "copy_ko": "..." },
    { "copy_ko": "..." },
    { "copy_ko": "..." }
  ]
}
"""


# ============================
# 🧾 Pydantic Schemas
# ============================

class UploadImageResponse(BaseModel):
    id: int
    original_filename: str

    
class TranslateRequest(BaseModel):
    description_kr: str


class TranslateResponse(BaseModel):
    id: int
    description_kr: str
    description_en: str


class GenerateCopyRequest(BaseModel):
    description_id: int
    strategy_id: int
    strategy_name: str
    product_name: str
    foreground_analysis: Optional[str] = ""


class AdCopyVariant(BaseModel):
    id: int
    copy_ko: str


class GenerateCopyResponse(BaseModel):
    description_id: int
    strategy_id: int
    strategy_name: str
    product_name: str
    variants: List[AdCopyVariant]
    image_id: Optional[int] = None

# ============================
# 🔁 DB Dependency
# ============================

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============================
# 🤖 GPT Call Helpers
# ============================

def translate_description(description_kr: str) -> str:
    """Korean → natural English via GPT."""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
            {"role": "user", "content": description_kr},
        ],
        temperature=0.3,
    )
    data = json.loads(resp.choices[0].message.content)
    return data["translation_en"].strip()


def build_copy_user_prompt(
    strategy_id: int,
    strategy_name: str,
    product_name: str,
    desc_kr: str,
    desc_en: str,
    fg_analysis: str,
) -> str:
    return f"""
strategy_id: {strategy_id}
strategy_name: {strategy_name}

product_name:
{product_name}

user_description_kr:
{desc_kr}

user_description_en:
{desc_en}

foreground_analysis:
{fg_analysis}

Please generate 3 Korean ad copy variants following the system rules.
"""


def generate_copy_variants_gpt(
    strategy_id: int,
    strategy_name: str,
    product_name: str,
    desc_kr: str,
    desc_en: str,
    fg_analysis: str,
) -> dict:
    user_prompt = build_copy_user_prompt(
        strategy_id, strategy_name, product_name, desc_kr, desc_en, fg_analysis
    )

    resp = client.chat.completions.create(
        model="gpt-4.1",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": GPT_COPY_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
    )
    return json.loads(resp.choices[0].message.content)


# ============================
# 🚀 FastAPI App
# ============================

app = FastAPI(
    title="Ad Copy Generator API",
    description="User descriptions to generate ad copies using GPT.",
    version="1.0.0",
    lifespan=lifespan,
)

# allow Expo dev URLs, adjust as needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev; tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------
# 🩺 Health Check
# ---------------------------------
@app.get("/healthz", status_code=200)
def health_check():
    return {"status": "ok"}


# ---------------------------------
# 1️⃣ POST /translate-description
# ---------------------------------
@app.post("/translate-description", response_model=TranslateResponse)
def translate_and_store(req: TranslateRequest, db: Session = Depends(get_db)):

    if not req.description_kr.strip():
        raise HTTPException(status_code=400, detail="description_kr is empty")

    try:
        description_en = translate_description(req.description_kr)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {e}")

    record = UserDescription(
        description_kr=req.description_kr.strip(),
        description_en=description_en,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return TranslateResponse(
        id=record.id,
        description_kr=record.description_kr,
        description_en=record.description_en,
    )


# ---------------------------------
# 2️⃣ POST /generate-copy-variants
# ---------------------------------
@app.post("/generate-copy-variants", response_model=GenerateCopyResponse)
def generate_copy_variants(req: GenerateCopyRequest, db: Session = Depends(get_db)):

    # 1) Fetch description from DB
    desc: UserDescription = db.query(UserDescription).filter(
        UserDescription.id == req.description_id
    ).first()

    if not desc:
        raise HTTPException(
            status_code=404,
            detail=f"UserDescription id={req.description_id} not found",
        )

    # 2) Call GPT to generate 3 Korean variants
    try:
        gpt_result = generate_copy_variants_gpt(
            strategy_id=req.strategy_id,
            strategy_name=req.strategy_name,
            product_name=req.product_name,
            desc_kr=desc.description_kr,
            desc_en=desc.description_en,
            fg_analysis=req.foreground_analysis or "",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GPT copy generation failed: {e}")

    variants_data = gpt_result.get("variants", [])
    if not variants_data or len(variants_data) != 3:
        raise HTTPException(
            status_code=500,
            detail="GPT result does not contain exactly 3 variants",
        )

    # 3) Store in DB
    created_variants = []
    new_ad_copies = []
    for v in variants_data:
        copy_ko = v.get("copy_ko", "").strip()
        if not copy_ko:
            continue

       rec = AdCopy(
            description_id=desc.id,
            strategy_id=req.strategy_id,
            strategy_name=req.strategy_name,
            product_name=req.product_name,
            copy_ko=copy_ko,
            image_id=req.image_id,  # may be None
        )

        db.add(rec)
        new_ad_copies.append(rec)

    if not new_ad_copies:
        raise HTTPException(
            status_code=500, detail="No valid variants were created from GPT result"
        )

    db.commit()

    for rec in new_ad_copies:
        db.refresh(rec)
        created_variants.append(
            AdCopyVariant(
                id=rec.id,
                copy_ko=rec.copy_ko,
            )
        )

    return GenerateCopyResponse(
        description_id=desc.id,
        strategy_id=req.strategy_id,
        strategy_name=req.strategy_name,
        product_name=req.product_name,
        variants=created_variants,
    )

# ---------------------------------
# 3 POST /upload-image
# ---------------------------------
@app.post("/upload-image", response_model=UploadImageResponse)
async def upload_image(file: UploadFile = File(...)):
    db: Session = next(get_db())

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Generate a unique filename
    ext = os.path.splitext(file.filename)[1].lower()
    unique_name = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(MEDIA_ROOT, unique_name)

    # Save file to disk
    content = await file.read()
    try:
        with open(save_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Store metadata in DB
    record = UploadedImage(
        file_path=save_path,
        original_filename=file.filename,
        content_type=file.content_type or "",
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return UploadImageResponse(
        id=record.id,
        original_filename=record.original_filename,
    )
