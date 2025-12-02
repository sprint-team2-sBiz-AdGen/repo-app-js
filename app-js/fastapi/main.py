import os
import uuid
import json
import asyncio
from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException, Request
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Any, Optional

from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# --- Use Async SQLAlchemy components ---
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID, JSONB 
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy.future import select

# ============================
# 🔐 Environment & Config
# ============================
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://feedlyai:feedlyai_dev_password_74154@localhost:5432/feedlyai")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY: raise RuntimeError("OPENAI_API_KEY not set")
client = OpenAI(api_key=OPENAI_API_KEY)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MEDIA_ROOT = os.path.join(PROJECT_ROOT, "media", "uploads")
os.makedirs(MEDIA_ROOT, exist_ok=True)

# ============================
# 🧠 CORE 8-STRATEGY PROMPT ENGINEERING
# ============================
TRANSLATION_SYSTEM_PROMPT = "You are a professional marketing copy translator. Your task is to translate a Korean description of a food menu item into natural, marketing-friendly English. Do NOT translate word-for-word. Output JSON ONLY: {\"translation_en\": \"...\"}"
GPT_COPY_SYSTEM_PROMPT = """
You are Feedly AI, an AI assistant that generates Korean Instagram ad copy for small F&B business owners. You will be given a strategy and a description. Your task is to generate THREE distinct Korean ad copy options.

Strategy tones:
  1. Hero Dish Focus — emphasize visual deliciousness & texture.
  2. Seasonal / Limited — urgency + seasonal mood.
  3. Behind-the-Scenes — sincerity, craftsmanship.
  4. Lifestyle — cozy, everyday scene.
  5. UGC / Social Proof — authentic, casual customer vibe.
  6. Minimalist Branding — clean, premium, one short sentence.
  7. Emotion / Comfort — warm, nostalgic.
  8. Retro / Vintage — storytelling, old-days atmosphere.

Rules:
- Output ONLY Korean copy for each variant.
- Each of the 3 variants must feel clearly different.
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
ENG_TO_KOR_TRANSLATION_PROMPT = """
You are a professional marketing copy translator specializing in Instagram food ads. Translate each English ad copy variant into natural, appealing Korean. Preserve marketing tone. No hashtags. No emojis.
Output JSON ONLY:
{
  "variants": [
    {"copy_ko": "..."},
    {"copy_ko": "..."},
    {"copy_ko": "..."}
  ]
}
"""

# ============================
# 💾 SQLAlchemy Async Setup
# ============================
Base = declarative_base()
engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

# --- Official, Unified Database Models (Using UUID as per schema) ---
class Job(Base):
    __tablename__ = 'jobs'
    job_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # optional: add tenant_id/store_id if present in DB
    # tenant_id = Column(String)
    # store_id = Column(UUID(as_uuid=True), ForeignKey('stores.store_id'))
    status = Column(String, default='queued')
    current_step = Column(String)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    inputs = relationship("JobInput", back_populates="job", uselist=False)

class ImageAsset(Base):
    __tablename__ = 'image_assets'
    image_asset_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    image_url = Column(String, nullable=False)
    # optional: add image_type if exists
    # image_type = Column(String)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

class JobInput(Base):
    __tablename__ = 'job_inputs'
    pk = Column(Integer, primary_key=True, autoincrement=True)  # matches SERIAL pk
    job_id = Column(UUID(as_uuid=True), ForeignKey('jobs.job_id'), nullable=False)
    img_asset_id = Column(UUID(as_uuid=True), ForeignKey('image_assets.image_asset_id'), nullable=False)  # exact column name
    desc_kor = Column(Text)
    desc_eng = Column(Text)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    job = relationship("Job", back_populates="inputs")

class TxtAdCopyGeneration(Base):
    __tablename__ = 'txt_ad_copy_generations'
    ad_copy_gen_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey('jobs.job_id'))
    strategy_id = Column(String)
    strategy_name = Column(String)
    ad_copy_variants = Column(JSONB)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

# --- Pydantic Models & Form Dependency Class ---
class JobCreateRequest(BaseModel):
    description: str

class AdCopyEngRequest(BaseModel):
    job_id: str
    strategy_id: int
    strategy_name: str

class KorToEngRequest(BaseModel):
    job_id: str

class AdCopyKorRequest(BaseModel):
    job_id: str
    strategy_id: int
    strategy_name: str

# ============================
# 🚀 FastAPI App & Helpers
# ============================
async def get_db():
    async with AsyncSessionLocal() as session: yield session

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Application startup...")
    yield
    print("👋 Application shutdown.")

app = FastAPI(title="Feedly AI - 8 Strategy API", version="4.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

async def call_gpt(system_prompt: str, user_prompt: str) -> dict:
    try:
        resp = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4o", response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.7,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")

# ============================
# 🔗 API Endpoints
# ============================
@app.get("/", include_in_schema=False)
async def root(): return {"message": "Feedly AI API is running"}

# Replace the existing create_job endpoint with this more tolerant version
@app.post("/api/v1/jobs/create")
async def create_job(
    image: UploadFile = File(...),
    request_field: Optional[str] = Form(None, alias="request"),
    description_field: Optional[str] = Form(None, alias="description"),
    db: AsyncSession = Depends(get_db)
):
    # Log raw inputs to confirm parser
    print("DEBUG request_field:", request_field)
    print("DEBUG description_field:", description_field)
    if image is None:
        raise HTTPException(status_code=422, detail="Missing file field 'image'")

    # Prefer 'request' JSON, fallback to plain 'description'
    raw = request_field or description_field
    if not raw:
        raise HTTPException(status_code=422, detail="Missing form field: 'request' or 'description'")
    try:
        data = json.loads(raw) if raw.strip().startswith("{") else {"description": raw}
        description = data["description"]
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid description payload: {e}")

    # Save image
    filename = f"{uuid.uuid4().hex}{os.path.splitext(image.filename)[1]}"
    path = os.path.join(MEDIA_ROOT, filename)
    try:
        with open(path, "wb") as f:
            f.write(await image.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save image: {e}")

    # DB writes (ensure column names match schema)
    new_image = ImageAsset(image_url=path)
    new_job = Job(status="running")
    db.add_all([new_image, new_job])
    await db.flush()

    new_input = JobInput(
        job_id=new_job.job_id,
        img_asset_id=new_image.image_asset_id,  # matches DB column name
        desc_kor=description,
    )
    db.add(new_input)
    await db.commit()

    return {"job_id": str(new_job.job_id), "status": "job_created"}

@app.post("/api/js/gpt/kor-to-eng")
async def gpt_kor_to_eng(req: KorToEngRequest, db: AsyncSession = Depends(get_db)):
    job_id = uuid.UUID(req.job_id)
    job_input = (await db.execute(select(JobInput).filter(JobInput.job_id == job_id))).scalar_one()
    
    gpt_result = await call_gpt(TRANSLATION_SYSTEM_PROMPT, job_input.desc_kor)
    job_input.desc_eng = gpt_result.get("translation_en", "Translation failed.")
    
    await db.commit()
    return {"job_id": str(job_id), "desc_eng": job_input.desc_eng}

@app.post("/api/js/gpt/ad-copy-eng")
async def gpt_ad_copy_eng(req: AdCopyEngRequest, db: AsyncSession = Depends(get_db)):
    job_id = uuid.UUID(req.job_id)
    job_input = (await db.execute(select(JobInput).filter(JobInput.job_id == job_id))).scalar_one()
    if not job_input.desc_eng: raise HTTPException(404, "English description not found.")

    user_prompt = f"strategy_id: {req.strategy_id}\nstrategy_name: {req.strategy_name}\nuser_description_kr: {job_input.desc_kor}\nuser_description_en: {job_input.desc_eng}"
    gpt_result = await call_gpt(GPT_COPY_SYSTEM_PROMPT, user_prompt)
    variants = gpt_result.get("variants", [])

    gen_record = TxtAdCopyGeneration(
        job_id=job_id,
        strategy_id=str(req.strategy_id),
        strategy_name=req.strategy_name,
        ad_copy_variants=variants
    )
    db.add(gen_record)
    await db.commit()
    return {"job_id": str(job_id), "variants": variants}

@app.post("/api/js/gpt/ad-copy-kor")
async def gpt_ad_copy_kor(req: AdCopyKorRequest, db: AsyncSession = Depends(get_db)):
    job_id = uuid.UUID(req.job_id)
    # 1. Get latest English variants for this strategy
    eng_gen_result = await db.execute(
        select(TxtAdCopyGeneration)
        .filter(
            TxtAdCopyGeneration.job_id == job_id,
            TxtAdCopyGeneration.strategy_id == str(req.strategy_id),
            TxtAdCopyGeneration.strategy_name == req.strategy_name
        )
        .order_by(TxtAdCopyGeneration.created_at.desc())
    )
    eng_gen = eng_gen_result.scalars().first()
    if not eng_gen or not eng_gen.ad_copy_variants:
        raise HTTPException(status_code=404, detail="English ad copy variants not found for translation.")
    english_variants_json = json.dumps(eng_gen.ad_copy_variants)
    # 2. Translate to Korean
    gpt_result = await call_gpt(ENG_TO_KOR_TRANSLATION_PROMPT, english_variants_json)
    variants_ko = gpt_result.get("variants", [])
    if not variants_ko:
        raise HTTPException(status_code=500, detail="Translation failed.")
    # 3. Persist Korean variants
    ko_record = TxtAdCopyGeneration(
        job_id=job_id,
        strategy_id=str(req.strategy_id),
        strategy_name=req.strategy_name,
        ad_copy_variants=variants_ko
    )
    db.add(ko_record)
    await db.commit()
    return {"job_id": str(job_id), "variants_ko": variants_ko}
