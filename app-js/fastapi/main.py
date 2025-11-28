import os
import json
import uuid
import asyncio
from fastapi import UploadFile, File
from contextlib import asynccontextmanager
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Any

# --- Use Async SQLAlchemy components ---
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy.future import select

from openai import OpenAI

# ============================
# 🔐 Environment & Config
# ============================

load_dotenv()

POSTGRES_USER = os.getenv("POSTGRES_USER", "feedlyai")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "feedlyai_dev_password_74154")
POSTGRES_DB = os.getenv("POSTGRES_DB", "feedlyai")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

# --- Use the asyncpg driver ---
DATABASE_URL = f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")
client = OpenAI(api_key=OPENAI_API_KEY)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MEDIA_ROOT = os.path.join(PROJECT_ROOT, "media", "uploads")
MEDIA_ROOT = os.getenv("MEDIA_ROOT", DEFAULT_MEDIA_ROOT)
os.makedirs(MEDIA_ROOT, exist_ok=True)

# ============================
# 💾 SQLAlchemy Async Setup
# ============================

Base = declarative_base()
engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

# --- Models (Unchanged) ---
class UserDescription(Base):
    __tablename__ = "user_descriptions"
    id = Column(Integer, primary_key=True, index=True)
    description_kr = Column(Text, nullable=False)
    description_en = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    generations = relationship("AdCopyGeneration", back_populates="description")

class UploadedImage(Base):
    __tablename__ = "uploaded_images"
    id = Column(Integer, primary_key=True, index=True)
    file_path = Column(String(512), nullable=False)
    original_filename = Column(String(255), nullable=False)
    content_type = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    generations = relationship("AdCopyGeneration", back_populates="image")

class AdCopyGeneration(Base):
    __tablename__ = "ad_copy_generations"
    id = Column(Integer, primary_key=True, index=True)
    description_id = Column(Integer, ForeignKey("user_descriptions.id"), nullable=False)
    strategy_id = Column(Integer, nullable=False)
    strategy_name = Column(String(100), nullable=False)
    image_id = Column(Integer, ForeignKey("uploaded_images.id"), nullable=True)
    variants = Column(JSONB, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    description = relationship("UserDescription", back_populates="generations")
    image = relationship("UploadedImage", back_populates="generations")

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✅ Database tables created successfully.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Application startup...")
    await init_db()
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
# --- Pydantic Models ---
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
    image_id: Optional[int] = None
    foreground_analysis: Optional[str] = ""

# --- FIX: Consolidated Response Models ---
class GenerateCopyResponse(BaseModel):
    generation_id: int
    variants: List[Any]

class GenerationResult(BaseModel):
    id: int
    strategy_name: str
    variants: List[Any]
    created_at: datetime

    class Config:
        from_attributes = True

# ============================
# 🔁 DB Dependency
# ============================

async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session


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
    product_name: Optional[str],  # <-- MAKE THIS OPTIONAL
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
    product_name: Optional[str], # <-- FIX 3: Accept product_name here
    desc_kr: str,
    desc_en: str,
    fg_analysis: str,
) -> dict:
    user_prompt = build_copy_user_prompt(
        strategy_id, strategy_name, product_name, desc_kr, desc_en, fg_analysis
    )

    resp = client.chat.completions.create(
        model="gpt-4o",  # <-- FIX: Use a valid model name
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
async def health_check():
    return {"status": "ok"}


# ---------------------------------
# 1️⃣ POST /translate-description
# ---------------------------------
@app.post("/translate-description", response_model=TranslateResponse)
async def translate_and_store(req: TranslateRequest, db: AsyncSession = Depends(get_db)):

    if not req.description_kr.strip():
        raise HTTPException(status_code=400, detail="description_kr is empty")

    try:
        # Run synchronous OpenAI call in a separate thread to avoid blocking
        description_en = await asyncio.to_thread(translate_description, req.description_kr)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {e}")

    record = UserDescription(
        description_kr=req.description_kr.strip(),
        description_en=description_en,
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)

    return record


# ---------------------------------
# 2️⃣ POST /generate-copy-variants
# ---------------------------------
@app.post("/generate-copy-variants", response_model=GenerateCopyResponse)
async def generate_copy_variants(req: GenerateCopyRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(UserDescription).filter(UserDescription.id == req.description_id))
    desc = result.scalar_one_or_none()
    if not desc:
        raise HTTPException(status_code=404, detail=f"Description with id {req.description_id} not found")
    
    variants_data = await asyncio.to_thread(
        generate_copy_variants_gpt,
        strategy_id=req.strategy_id,
        strategy_name=req.strategy_name,
        product_name=None,
        desc_kr=desc.description_kr,
        desc_en=desc.description_en,
        fg_analysis=req.foreground_analysis,
    )
    
    valid_variants = variants_data.get("variants", [])
    if not valid_variants:
        raise HTTPException(status_code=500, detail="No valid variants were created from GPT result")
    
    generation_record = AdCopyGeneration(
        description_id=req.description_id,
        strategy_id=req.strategy_id,
        strategy_name=req.strategy_name,
        image_id=req.image_id,
        variants=valid_variants,
    )
    db.add(generation_record)
    await db.commit()
    await db.refresh(generation_record)

    # --- FIX: Construct the correct response object ---
    # The response model expects a 'generation_id' field, but our DB model has 'id'.
    # We create a dictionary that matches the 'GenerateCopyResponse' model.
    return {
        "generation_id": generation_record.id,
        "variants": generation_record.variants
    }

# ---------------------------------
# 3 POST /upload-image
# ---------------------------------
@app.post("/upload-image", response_model=UploadImageResponse)
async def upload_image(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    ext = os.path.splitext(file.filename)[1].lower()
    unique_name = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(MEDIA_ROOT, unique_name)
    content = await file.read()
    
    try:
        # --- FIX: Define a helper function for the blocking I/O ---
        def save_file_sync():
            with open(save_path, "wb") as f:
                f.write(content)
        
        # Run the helper function in a separate thread
        await asyncio.to_thread(save_file_sync)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
         
    record = UploadedImage(file_path=save_path, original_filename=file.filename, content_type=file.content_type or "")
    db.add(record)
    await db.commit()
    await db.refresh(record)
    return record

# --- ADD THIS NEW ENDPOINT ---
# ---------------------------------
# 4️⃣ GET /generations
# ---------------------------------
@app.get("/generations", response_model=List[GenerationResult])
async def get_generations(db: AsyncSession = Depends(get_db), limit: int = 10):
    result = await db.execute(select(AdCopyGeneration).order_by(AdCopyGeneration.created_at.desc()).limit(limit))
    return result.scalars().all()

# --- ADD THIS NEW ENDPOINT ---
# ---------------------------------
# 5️⃣ GET /generations/{generation_id}
# ---------------------------------
@app.get("/generations/{generation_id}", response_model=GenerationResult)
async def get_generation_by_id(generation_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(AdCopyGeneration).filter(AdCopyGeneration.id == generation_id))
    generation = result.scalar_one_or_none()
    if not generation:
        raise HTTPException(status_code=404, detail="Generation not found")
    return generation
