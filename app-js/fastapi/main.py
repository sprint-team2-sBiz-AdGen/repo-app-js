# updated at: 2025-12-02
# version: 1.7.0

import os
import uuid
import json
import asyncio
from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.requests import Request as StarletteRequest
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Any, Optional, Tuple

from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from fastapi.staticfiles import StaticFiles

from openai import OpenAI

# --- Use Async SQLAlchemy components ---
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Integer, Float, Boolean, text, func
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

# User-based defaults (override via .env)
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID")            # users.user_id UUID for image_assets.creator_id

client = OpenAI(api_key=OPENAI_API_KEY)

# --- CORRECTED ASSET PATHS ---
# This is the directory on your server where the final images are stored, as per your screenshot.
ASSETS_ROOT = "/opt/feedlyai/assets"
# This is the URL prefix under which the assets will be served.
ASSETS_URL_PREFIX = "/assets"

# This media root is for temporary uploads, not the final assets.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MEDIA_ROOT = os.path.join(PROJECT_ROOT, "media", "uploads")
os.makedirs(MEDIA_ROOT, exist_ok=True)

# ============================
# 🧠 CORE 8-STRATEGY PROMPT ENGINEERING
# ============================
TRANSLATION_SYSTEM_PROMPT = "You are a professional marketing copy translator. Your task is to translate a Korean description of a food menu item into natural, marketing-friendly English. Do NOT translate word-for-word. Output JSON ONLY: {\"translation_en\": \"...\"}"

# --- FIX: Simplified to generate ONE plain text English ad copy ---
GPT_COPY_SYSTEM_PROMPT = """
You are Feedly AI, an AI assistant that generates Instagram ad copy. You will be given an English description.
Your task is to generate ONE compelling English ad copy option based on one of the following strategies: Hero Dish Focus, Seasonal, Behind-the-Scenes, or Lifestyle.
- Output ONLY the single ad copy text.
- Do NOT include strategy names, hashtags, emojis, or any other text.
- Do NOT use JSON.
"""

# --- FIX: Simplified to generate ONE plain text Korean ad copy ---
ENG_TO_KOR_TRANSLATION_PROMPT = """
You are a professional marketing copy translator specializing in Instagram food ads.
Translate the given English ad copy into a single, natural, and appealing Korean sentence.
- Output ONLY the translated Korean text.
- Do NOT include hashtags, emojis, or any other text.
- Do NOT use JSON.
"""

# ============================
# 💾 SQLAlchemy Async Setup
# ============================
Base = declarative_base()
engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

# --- Official, Unified Database Models (Using UUID as per schema) ---
class Tenant(Base):
    __tablename__ = 'tenants'
    tenant_id = Column(String, primary_key=True)
    display_name = Column(String)
    uid = Column(String, unique=True)
    pk = Column(Integer, server_default=text("nextval('tenants_pk_seq'::regclass)"), nullable=False)  # SERIAL with server default
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class User(Base):
    __tablename__ = 'users'
    user_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    uid = Column(String(255), unique=True)
    pk = Column(Integer, server_default=text("nextval('users_pk_seq'::regclass)"), nullable=False)  # SERIAL with server default
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class ToneStyle(Base):
    __tablename__ = 'tone_styles'
    tone_style_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    code = Column(Text)
    kor_name = Column(Text)
    eng_name = Column(Text)
    description = Column(Text)
    pk = Column(Integer, server_default=text("nextval('tone_styles_pk_seq'::regclass)"), nullable=False)  # SERIAL with server default
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class Store(Base):
    __tablename__ = 'stores'
    store_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.user_id'), nullable=True)
    image_id = Column(UUID(as_uuid=True), ForeignKey('image_assets.image_asset_id'), nullable=True)
    title = Column(String(500))
    body = Column(Text)
    store_category = Column(Text)
    auto_scoring_flag = Column(Boolean, default=False)
    pk = Column(Integer, server_default=text("nextval('stores_pk_seq'::regclass)"), nullable=False)  # SERIAL with server default
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class Job(Base):
    __tablename__ = 'jobs'
    job_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(String, ForeignKey('tenants.tenant_id'), nullable=True)
    store_id = Column(UUID(as_uuid=True), nullable=True) 
    status = Column(String, default='queued')
    version = Column(String, nullable=True)
    current_step = Column(String, default='job_created')
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    retry_count = Column(Integer, default=0)
    inputs = relationship("JobInput", back_populates="job", uselist=False)

class ImageAsset(Base):
    __tablename__ = 'image_assets'
    image_asset_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    image_type = Column(String, default='original')
    image_url = Column(Text, nullable=False)
    mask_url = Column(Text, nullable=True)
    width = Column(Integer, default=1080)
    height = Column(Integer, default=1080)
    creator_id = Column(UUID(as_uuid=True), ForeignKey('users.user_id'), nullable=True)
    tenant_id = Column(String, ForeignKey('tenants.tenant_id'), nullable=True)  # Remove default, set explicitly
    pk = Column(Integer, server_default=text("nextval('image_assets_pk_seq'::regclass)"), nullable=False)  # SERIAL with server default
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    job_id = Column(UUID(as_uuid=True), ForeignKey('jobs.job_id'), nullable=False, primary_key=True)

class JobInput(Base):
    __tablename__ = 'job_inputs'
    job_id = Column(UUID(as_uuid=True), ForeignKey('jobs.job_id'), nullable=False, primary_key=True)  # PK, FK
    img_asset_id = Column(UUID(as_uuid=True), ForeignKey('image_assets.image_asset_id'), nullable=False)  # exact column name
    tone_style_id = Column(UUID(as_uuid=True), ForeignKey('tone_styles.tone_style_id'), nullable=True)
    desc_kor = Column(Text)
    desc_eng = Column(Text)
    pk = Column(Integer, server_default=text("nextval('job_inputs_pk_seq'::regclass)"), nullable=False)  # SERIAL with server default
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    job = relationship("Job", back_populates="inputs")

class LLMModel(Base):
    __tablename__ = 'llm_models'
    llm_model_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String(255), nullable=False)
    model_version = Column(String(255))
    provider = Column(String(255), nullable=False)
    default_temperature = Column(Float)
    default_max_tokens = Column(Integer)
    is_active = Column(String(10), default='true')

class LLMTrace(Base):
    __tablename__ = 'llm_traces'
    llm_trace_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey('jobs.job_id'))
    provider = Column(String)
    llm_model_id = Column(UUID(as_uuid=True), ForeignKey('llm_models.llm_model_id'), nullable=True)
    tone_style_id = Column(UUID(as_uuid=True), nullable=True)
    enhanced_img_id = Column(UUID(as_uuid=True), nullable=True)
    prompt_id = Column(UUID(as_uuid=True), nullable=True)
    operation_type = Column(String)
    request = Column(JSONB)
    response = Column(JSONB)
    latency_ms = Column(Float)
    prompt_tokens = Column(Integer)
    completion_tokens = Column(Integer)
    total_tokens = Column(Integer)
    token_usage = Column(JSONB)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow)

class TxtAdCopyGeneration(Base):
    __tablename__ = 'txt_ad_copy_generations'
    ad_copy_gen_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey('jobs.job_id'), nullable=False)
    llm_trace_id = Column(UUID(as_uuid=True), ForeignKey('llm_traces.llm_trace_id'), nullable=True)
    generation_stage = Column(String, nullable=False)  # 'kor_to_eng', 'ad_copy_eng', 'ad_copy_kor'
    ad_copy_kor = Column(Text)
    ad_copy_eng = Column(Text)
    refined_ad_copy_eng = Column(Text)
    status = Column(String, default='queued')
    pk = Column(Integer, server_default=text("nextval('txt_ad_copy_generations_pk_seq'::regclass)"), nullable=False)  # SERIAL with server default
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow)

# --- Pydantic Models & Form Dependency Class ---
class JobCreateRequest(BaseModel):
    description: str

class KorToEngRequest(BaseModel):
    job_id: str
    tenant_id: Optional[str] = None

class AdCopyEngRequest(BaseModel):
    job_id: str
    tenant_id: Optional[str] = None

class AdCopyKorRequest(BaseModel):
    job_id: str
    tenant_id: Optional[str] = None

# --- DEFINITIVE FIX for ORM models ---
class JobVariant(Base):
    __tablename__ = 'jobs_variants'
    job_variants_id = Column('job_variants_id', UUID(as_uuid=True), primary_key=True)
    job_id = Column(UUID(as_uuid=True), ForeignKey('jobs.job_id'))
    img_asset_id = Column(UUID(as_uuid=True), ForeignKey('image_assets.image_asset_id'))
    overlaid_img_asset_id = Column(UUID(as_uuid=True), ForeignKey('image_assets.image_asset_id'), nullable=True)
    creation_order = Column(Integer)
    status = Column(String)
    current_step = Column(String)
    retry_count = Column(Integer)
    selected = Column(Boolean)
    overlaid_image = relationship("ImageAsset", foreign_keys=[overlaid_img_asset_id])

class InstagramFeed(Base):
    __tablename__ = 'instagram_feeds'
    instagram_feed_id = Column('instagram_feed_id', UUID(as_uuid=True), primary_key=True)
    job_id = Column(UUID(as_uuid=True), ForeignKey('jobs.job_id'))
    overlay_id = Column(UUID(as_uuid=True), nullable=True)
    tenant_id = Column(String, nullable=True)
    refined_ad_copy_eng = Column(Text, nullable=True)
    tone_style = Column(String, nullable=True)
    product_description = Column(Text, nullable=True)
    gpt_prompt = Column(Text, nullable=True)
    instagram_ad_copy = Column(Text)
    hashtags = Column(Text, nullable=True)
    latency_ms = Column(Float, nullable=True)
    used_temperature = Column(Float, nullable=True)
    used_max_tokens = Column(Integer, nullable=True)
    llm_trace_id = Column(UUID(as_uuid=True), nullable=True)
    ad_copy_kor = Column(Text, nullable=True)
    pk = Column(Integer, server_default=text("nextval('instagram_feeds_pk_seq'::regclass)"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
# --- Pydantic response models ---
class JobResultImage(BaseModel):
    image_url: str
    class Config:
        from_attributes = True

class JobResult(BaseModel):
    images: List[JobResultImage]
    instagram_ad_copy: Optional[str] = None
    hashtags: Optional[str] = None

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

# --- CORRECTED STATIC FILE MOUNTING ---
# Serve the final assets from the /opt/feedlyai/assets directory.
app.mount(ASSETS_URL_PREFIX, StaticFiles(directory=ASSETS_ROOT), name="assets")

# This mount for /media can remain for temporary uploads if needed.
app.mount("/media", StaticFiles(directory=MEDIA_ROOT), name="media")

# Add exception handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    print("=" * 50)
    print("VALIDATION ERROR:")
    print(f"Request URL: {request.url}")
    print(f"Request method: {request.method}")
    try:
        error_list = exc.errors()
        print(f"Error details: {error_list}")
    except Exception as e:
        print(f"Error getting error details: {e}")
        import traceback
        traceback.print_exc()
    try:
        body = await request.body()
        print(f"Request body (first 500 chars): {body[:500]}")
    except Exception as e:
        print(f"Could not read body: {e}")
    print("=" * 50)
    # Convert errors to JSON-serializable format
    errors = []
    try:
        for error in exc.errors():
            error_dict = {
                "type": str(error.get("type", "")),
                "loc": list(error.get("loc", [])),
                "msg": str(error.get("msg", "")),
            }
            # Handle input field carefully - it might contain non-serializable objects
            input_val = error.get("input", "")
            try:
                error_dict["input"] = str(input_val)
            except Exception:
                error_dict["input"] = f"<non-serializable: {type(input_val)}>"
            errors.append(error_dict)
    except Exception as e:
        print(f"Error processing validation errors: {e}")
        import traceback
        traceback.print_exc()
        errors = [{"error": "Failed to process validation errors", "detail": str(e)}]
    
    return JSONResponse(
        status_code=422,
        content={"detail": errors}
    )

# Add general exception handler to catch all errors
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    print("=" * 50)
    print("GENERAL EXCEPTION:")
    print(f"Request URL: {request.url}")
    print(f"Request method: {request.method}")
    print(f"Exception type: {type(exc).__name__}")
    print(f"Exception message: {str(exc)}")
    import traceback
    traceback.print_exc()
    print("=" * 50)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {type(exc).__name__}: {str(exc)}"}
    )

async def call_gpt(system_prompt: str, user_prompt: str, model_name: str = "gpt-4o-mini", expect_json: bool = False) -> Tuple[Any, dict]:
    """
    Call GPT API and return (result, metadata).
    If expect_json is True, result is a dict.
    If expect_json is False, result is a string.
    """
    import time
    start_time = time.time()
    try:
        request_params = {
            "model": model_name,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            "temperature": 0.7,
        }
        if expect_json:
            request_params["response_format"] = {"type": "json_object"}

        resp = await asyncio.to_thread(
            client.chat.completions.create,
            **request_params
        )
        latency_ms = (time.time() - start_time) * 1000
        
        raw_content = resp.choices[0].message.content
        result = json.loads(raw_content) if expect_json else raw_content.strip()

        token_usage = {}
        if hasattr(resp, 'usage'):
            token_usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens
            }
        
        metadata = {
            "latency_ms": latency_ms, "token_usage": token_usage, "model": model_name,
            "request": {"system": system_prompt, "user": user_prompt}, "response": raw_content
        }
        return result, metadata
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")

async def get_or_create_llm_model(db: AsyncSession, model_name: str, provider: str = "openai") -> LLMModel:
    """Fetches an LLMModel by name or creates it if it doesn't exist."""
    result = await db.execute(select(LLMModel).filter_by(model_name=model_name))
    model = result.scalar_one_or_none()
    if not model:
        print(f"INFO: LLM model '{model_name}' not found, creating new entry.")
        model = LLMModel(model_name=model_name, provider=provider, is_active='true')
        db.add(model)
        await db.flush()
    return model

# ============================
# 🔗 API Endpoints
# ============================
@app.get("/", include_in_schema=False)
async def root(): return {"message": "Feedly AI API is running"}

# Replace the existing create_job endpoint with this more tolerant version
@app.post("/api/v1/jobs/create", status_code=200)
async def create_job(
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    # Log raw inputs to confirm parser
    print("=" * 50)
    print("DEBUG create_job called")
    print("DEBUG content-type:", request.headers.get("content-type"))
    
    # Try to parse as multipart form data
    try:
        form_data = await request.form()
        print("DEBUG form_data keys:", list(form_data.keys()))
        print("DEBUG form_data type:", type(form_data))
        
        # React Native FormData sends { uri, name, type } which gets stringified to "[object Object]"
        # We need to iterate through all form entries to find the actual file
        image = None
        request_field = None
        description_field = None
        
        # Iterate through all form entries
        for key in form_data.keys():
            value = form_data.get(key)
            print(f"DEBUG form key '{key}': type={type(value)}, has_read={hasattr(value, 'read') if value else False}")
            
            if key == "image":
                # Check if it's an UploadFile
                if hasattr(value, 'read') and hasattr(value, 'filename'):
                    image = value
                    print(f"DEBUG Found UploadFile: filename={value.filename}")
                else:
                    print(f"WARNING: 'image' field is not an UploadFile: {type(value)}, value={str(value)[:100]}")
            elif key == "image_base64":
                # React Native에서 base64로 인코딩된 이미지
                image_base64 = value
                image_name = form_data.get("image_name", "upload.jpg")
                print(f"DEBUG Found base64 image: name={image_name}, length={len(image_base64) if isinstance(image_base64, str) else 'N/A'}")
            elif key == "image_name":
                # 이미지 이름은 위에서 처리됨
                pass
            elif key == "request":
                request_field = value
            elif key == "description":
                description_field = value
        
        # If image is still None, try to get it from form_data._list (internal structure)
        if image is None:
            print("WARNING: Image not found in form_data.keys(), trying _list")
            if hasattr(form_data, '_list'):
                print(f"DEBUG _list type: {type(form_data._list)}, length: {len(form_data._list) if hasattr(form_data._list, '__len__') else 'N/A'}")
                for idx, item in enumerate(form_data._list):
                    print(f"DEBUG _list[{idx}]: type={type(item)}, value={str(item)[:100] if not hasattr(item, 'read') else 'UploadFile'}")
                    if isinstance(item, tuple) and len(item) >= 2:
                        key, value = item[0], item[1]
                        print(f"DEBUG _list item: key={key}, value_type={type(value)}")
                        if key == "image" and hasattr(value, 'read'):
                            image = value
                            print(f"DEBUG Found UploadFile in _list: filename={value.filename}")
                            break
        
        # Last resort: try to parse multipart body manually
        if image is None:
            print("WARNING: Image still not found, trying to parse multipart body manually")
            try:
                # Reset the request body stream
                body = await request.body()
                print(f"DEBUG body length: {len(body)}")
                
                # Try to use python-multipart to parse
                from multipart import parse_form_data
                import io
                
                # Parse the multipart data
                content_type = request.headers.get("content-type", "")
                boundary = content_type.split("boundary=")[-1] if "boundary=" in content_type else None
                
                if boundary:
                    print(f"DEBUG boundary: {boundary}")
                    # Create a file-like object from body
                    body_stream = io.BytesIO(body)
                    
                    # Parse multipart form data
                    fields, files = parse_form_data(body_stream, boundary.encode())
                    print(f"DEBUG parsed fields: {list(fields.keys())}")
                    print(f"DEBUG parsed files: {list(files.keys())}")
                    
                    if "image" in files:
                        file_info = files["image"]
                        print(f"DEBUG file_info: {file_info}")
                        # file_info is a tuple (filename, file_object)
                        if isinstance(file_info, tuple) and len(file_info) >= 2:
                            filename, file_obj = file_info[0], file_info[1]
                            # Read the file content
                            file_content = file_obj.read()
                            # Create a temporary UploadFile-like object
                            from fastapi import UploadFile as FastAPIUploadFile
                            from io import BytesIO
                            
                            image = FastAPIUploadFile(
                                filename=filename or "upload.jpg",
                                file=BytesIO(file_content)
                            )
                            print(f"DEBUG Created UploadFile from manual parsing: filename={filename}")
            except Exception as e:
                print(f"ERROR in manual multipart parsing: {e}")
                import traceback
                traceback.print_exc()
        
        # If image is still None, check if we have base64 image
        if image is None and 'image_base64' in locals() and image_base64:
            print("DEBUG Processing base64 image")
            try:
                import base64
                from io import BytesIO
                from fastapi import UploadFile as FastAPIUploadFile
                
                # Decode base64 image
                image_data = base64.b64decode(image_base64)
                image = FastAPIUploadFile(
                    filename=image_name if 'image_name' in locals() else "upload.jpg",
                    file=BytesIO(image_data)
                )
                print(f"DEBUG Created UploadFile from base64: filename={image.filename}")
            except Exception as e:
                print(f"ERROR decoding base64 image: {e}")
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=422, detail=f"Failed to decode base64 image: {str(e)}")
        
        if image is None:
            raise HTTPException(
                status_code=422,
                detail="Could not extract image file from FormData. React Native FormData may not be properly configured."
            )
        
        print("DEBUG image filename:", image.filename if hasattr(image, 'filename') else 'N/A')
        print("DEBUG image content_type:", image.content_type if hasattr(image, 'content_type') else 'N/A')
        
        # Get request or description field
        request_field = form_data.get("request")
        description_field = form_data.get("description")
        
        print("DEBUG request field:", request_field)
        print("DEBUG description field:", description_field)
        print("=" * 50)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR parsing form data: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=422, detail=f"Failed to parse form data: {str(e)}")
    
    # Prefer 'request' JSON, fallback to plain 'description'
    raw = request_field or description_field
    if not raw:
        raise HTTPException(status_code=422, detail="Missing form field: 'request' or 'description'")

    if not isinstance(raw, str):
        raw = str(raw)

    try:
        data = json.loads(raw) if raw.strip().startswith("{") else {"description": raw}
        description_text = (data.get("description") or raw).strip()
        # Pull optional tenant/store from payload (no defaults)
        tenant_id = data.get("tenant_id")
        store_id_uuid = None
        store_id_raw = data.get("store_id")
        if store_id_raw:
            try:
                store_id_uuid = uuid.UUID(str(store_id_raw))
            except Exception:
                store_id_uuid = None
        if not description_text:
            raise ValueError("Description is empty")
    except json.JSONDecodeError:
        description_text = raw.strip()
        tenant_id = None
        store_id_uuid = None
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid description payload: {e}")

    # --- FIX: Force a fixed tenant_id for all jobs ---
    tenant_id = "user_0b1bfa70-20a4-4807-a5b1-397e3c197ab8"
    print(f"DEBUG: Forcing fixed tenant_id: {tenant_id}")

    # Save image to the correct asset location
    # Path format: /assets/js/tenants/{tenant_id}/original/{year}/{month:02d}/{day:02d}/{filename}
    image_filename = image.filename if image.filename else 'upload.jpg'
    file_ext = os.path.splitext(image_filename)[1]
    unique_filename = f"{uuid.uuid4().hex}{file_ext}"
    
    # Set kind to "original" for create_job()
    kind = "original"
    
    # Get current date
    now = datetime.utcnow()
    year = now.year
    month = f"{now.month:02d}"
    day = f"{now.day:02d}"
    
    # Full directory path on the server's filesystem
    # e.g., /opt/feedlyai/assets/js/tenants/{tenant_id}/original/2025/12/03
    save_dir = os.path.join(ASSETS_ROOT, "js", "tenants", tenant_id, kind, str(year), month, day)
    os.makedirs(save_dir, exist_ok=True)
    
    # Final, full path to save the file on the server
    save_path = os.path.join(save_dir, unique_filename)
    
    # Save the image file
    try:
        image_content = await image.read()
        with open(save_path, "wb") as f:
            f.write(image_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save image: {e}")
    
    # This is the web-accessible URL path to store in the database
    # e.g., /assets/js/tenants/{tenant_id}/original/2025/12/03/some_uuid.jpg
    path = f"/assets/js/tenants/{tenant_id}/{kind}/{year}/{month}/{day}/{unique_filename}"

    # Reference existing records from tone_styles, users, tenants tables
    # 1. Get or create user based on DEFAULT_USER_ID
    user_obj = None
    creator_id = None
    if DEFAULT_USER_ID:
        try:
            user_uuid = uuid.UUID(str(DEFAULT_USER_ID))
            user_result = await db.execute(select(User).filter(User.user_id == user_uuid))
            user_obj = user_result.scalar_one_or_none()
            if user_obj:
                creator_id = user_obj.user_id
                print(f"DEBUG: Found user: {creator_id}")
            else:
                # User doesn't exist, create a new one
                print(f"INFO: User '{DEFAULT_USER_ID}' not found, creating new user")
                user_obj = User(
                    user_id=user_uuid,
                    uid=f"user_{user_uuid}"
                )
                db.add(user_obj)
                await db.flush()
                creator_id = user_obj.user_id
                print(f"DEBUG: Created new user: {creator_id}")
        except Exception as e:
            print(f"ERROR: Invalid DEFAULT_USER_ID '{DEFAULT_USER_ID}': {e}")
            raise HTTPException(status_code=400, detail=f"Invalid DEFAULT_USER_ID: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="DEFAULT_USER_ID is required but not set in environment variables")
    
    # 2. Get or create tenant based on user (user-specific tenant)
    tenant_obj = None
    if tenant_id:
        # If tenant_id is provided in request, use it
        tenant_result = await db.execute(select(Tenant).filter(Tenant.tenant_id == tenant_id))
        tenant_obj = tenant_result.scalar_one_or_none()
        if not tenant_obj:
            # Tenant doesn't exist, create a new one
            print(f"INFO: Tenant '{tenant_id}' not found in database, creating new tenant")
            tenant_obj = Tenant(
                tenant_id=tenant_id,
                display_name=f"Tenant {tenant_id}",
                uid=f"tenant_{tenant_id}"
            )
            db.add(tenant_obj)
            await db.flush()
            print(f"DEBUG: Created new tenant: {tenant_id}")
    # --- REMOVED: This 'else' block is no longer reachable because tenant_id is always set ---
    # else:
    #     # No tenant_id provided, create user-specific tenant
    #     user_tenant_id = f"user_{creator_id}"
    #     tenant_result = await db.execute(select(Tenant).filter(Tenant.tenant_id == user_tenant_id))
    #     tenant_obj = tenant_result.scalar_one_or_none()
    #     if tenant_obj:
    #         tenant_id = user_tenant_id
    #         print(f"DEBUG: Using existing user tenant: {tenant_id}")
    #     else:
    #         # Create user-specific tenant
    #         print(f"INFO: Creating user-specific tenant: {user_tenant_id}")
    #         tenant_obj = Tenant(
    #             tenant_id=user_tenant_id,
    #             display_name=f"User Tenant {creator_id}",
    #             uid=f"tenant_{user_tenant_id}"
    #         )
    #         db.add(tenant_obj)
    #         await db.flush()
    #         tenant_id = user_tenant_id
    #         print(f"DEBUG: Created user-specific tenant: {tenant_id}")
    
    # 3. Get or create store for the user
    store_obj = None
    # Check if store_id was provided in request
    if store_id_uuid:
        # If store_id is provided in request, verify it exists
        store_result = await db.execute(select(Store).filter(Store.store_id == store_id_uuid))
        store_obj = store_result.scalar_one_or_none()
        if not store_obj:
            print(f"WARNING: Store '{store_id_uuid}' not found in database, will create new one")
            store_id_uuid = None
    
    if not store_id_uuid:
        # No store_id provided or not found, find or create user-specific store
        store_result = await db.execute(select(Store).filter(Store.user_id == creator_id).limit(1))
        store_obj = store_result.scalar_one_or_none()
        if store_obj:
            store_id_uuid = store_obj.store_id
            print(f"DEBUG: Using existing user store: {store_id_uuid}")
        else:
            # Create user-specific store
            print(f"INFO: Creating user-specific store for user: {creator_id}")
            store_obj = Store(
                user_id=creator_id,
                title=f"Store for User {creator_id}",
                body=f"Default store for user {creator_id}",
                store_category="default"
            )
            db.add(store_obj)
            await db.flush()
            store_id_uuid = store_obj.store_id
            print(f"DEBUG: Created user-specific store: {store_id_uuid}")
    
    # 4. Get default tone_style (optional, can be None)
    tone_style_id = None
    try:
        # Try to get the first active tone_style (or any tone_style if no active flag)
        tone_style_result = await db.execute(select(ToneStyle).limit(1))
        tone_style_obj = tone_style_result.scalar_one_or_none()
        if tone_style_obj:
            tone_style_id = tone_style_obj.tone_style_id
            print(f"DEBUG: Using tone_style_id: {tone_style_id}")
        else:
            print("WARNING: No tone_style found in database, using None")
    except Exception as e:
        print(f"WARNING: Error fetching tone_style: {e}, using None")

    # DB writes (ensure column names match schema)
    new_job = Job(
        # The job is first created with this state
        status='done',
        current_step='job_created',
        tenant_id=tenant_id,
        store_id=store_id_uuid  # Use the store_id we found or created
    )
    db.add(new_job)
    await db.flush() # Flush to get the new_job.job_id

    new_image = ImageAsset(
        image_url=path,
        tenant_id=tenant_id,
        creator_id=creator_id,
        job_id=new_job.job_id  # FIX: Assign the job_id to the image asset
    )

    db.add(new_image)
    await db.flush() # Flush to get the new_image.image_asset_id

    new_input = JobInput(
        job_id=new_job.job_id,
        img_asset_id=new_image.image_asset_id,
        tone_style_id=tone_style_id,
        desc_kor=description_text  # use parsed description
    )
    db.add(new_input)

    # Immediately update the job to its next state before committing.
    # This is the final signal to the listener.
    new_job.current_step = 'user_img_input'
    new_job.status = 'done' 
    new_job.updated_at = datetime.utcnow()
    
    await db.commit()
    return {"job_id": str(new_job.job_id), "status": "job_created_and_inputs_submitted"}

@app.post("/api/js/gpt/kor-to-eng")
async def gpt_kor_to_eng(req: KorToEngRequest, db: AsyncSession = Depends(get_db)):
    """(FIXED) Translates Korean description and updates job_inputs.desc_eng."""
    try:
        job_id = uuid.UUID(req.job_id)
        model_name = "gpt-4o-mini"
        llm_model = await get_or_create_llm_model(db, model_name)
        
        job_input_res = await db.execute(select(JobInput).filter(JobInput.job_id == job_id))
        job_input = job_input_res.scalar_one_or_none()
        if not job_input or not job_input.desc_kor:
            raise HTTPException(status_code=404, detail="Job input with Korean description not found.")
        
        # This step still expects JSON, so expect_json=True
        gpt_result, metadata = await call_gpt(TRANSLATION_SYSTEM_PROMPT, job_input.desc_kor, model_name=model_name, expect_json=True)
        desc_eng = gpt_result.get("translation_en", "Translation failed.")
        
        llm_trace = LLMTrace(
            job_id = job_id,
            provider = 'openai', # Corrected provider name
            llm_model_id = llm_model.llm_model_id, # FIX: Use the ID from the fetched model object
            operation_type = 'kor_to_eng',
            request = {"system": TRANSLATION_SYSTEM_PROMPT, "user": job_input.desc_kor},
            response = {"content": desc_eng},
            latency_ms = metadata.get('latency_ms', 0),
            prompt_tokens = metadata.get('token_usage', {}).get('prompt_tokens'),
            completion_tokens = metadata.get('token_usage', {}).get('completion_tokens'),
            total_tokens = metadata.get('token_usage', {}).get('total_tokens'),
            token_usage = metadata.get('token_usage', {})
        )
        db.add(llm_trace)
        
        # 4. UPDATE the desc_eng field in the job_inputs table
        job_input.desc_eng = desc_eng
        job_input.updated_at = datetime.utcnow()
        
        # 5. Update job status to signal completion of this step
        job = await db.get(Job, job_id)
        if job:
            job.current_step = 'kor_to_eng'
            job.status = 'done'
            job.updated_at = datetime.utcnow()
        
        await db.commit()
        
        return {"job_id": str(job_id), "status": "done", "desc_eng": desc_eng}
    except Exception as e:
        print(f"ERROR in kor-to-eng: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/js/gpt/ad-copy-eng")
async def gpt_ad_copy_eng(req: AdCopyEngRequest, db: AsyncSession = Depends(get_db)):
    """(FIXED) Creates a single English ad copy and saves it to txt_ad_copy_generations."""
    job_id = uuid.UUID(req.job_id)
    model_name = "gpt-4o-mini"
    llm_model = await get_or_create_llm_model(db, model_name)
    
    job_input_res = await db.execute(select(JobInput).filter(JobInput.job_id == job_id))
    job_input = job_input_res.scalar_one_or_none()
    if not job_input or not job_input.desc_eng:
        raise HTTPException(status_code=404, detail="English description not found in job_inputs. Run kor-to-eng first.")
    
    # FIX: Call GPT expecting plain text (expect_json=False)
    user_prompt = f"Description (English): {job_input.desc_eng}"
    ad_copy_eng_text, metadata = await call_gpt(GPT_COPY_SYSTEM_PROMPT, user_prompt, model_name=model_name, expect_json=False)
    
    # Create LLM trace
    llm_trace = LLMTrace(
        job_id = job_id,
        provider = 'openai', # Corrected provider name
        llm_model_id = llm_model.llm_model_id, # FIX: Use the ID from the fetched model object
        operation_type = 'ad_copy_gen',
        request = metadata['request'],
        response = {"content": metadata['response']},
        latency_ms = metadata['latency_ms'],
        prompt_tokens = metadata['token_usage'].get('prompt_tokens'),
        completion_tokens = metadata['token_usage'].get('completion_tokens'),
        total_tokens = metadata['token_usage'].get('total_tokens'),
        token_usage = metadata['token_usage']
    )
    db.add(llm_trace)
    await db.flush()
    
    # UPSERT logic for txt_ad_copy_generations
    gen_res = await db.execute(select(TxtAdCopyGeneration).filter_by(job_id=job_id, generation_stage='ad_copy_eng'))
    ad_copy_gen = gen_res.scalar_one_or_none()
    
    if ad_copy_gen:
        ad_copy_gen.llm_trace_id = llm_trace.llm_trace_id
        ad_copy_gen.ad_copy_eng = ad_copy_eng_text # FIX: Save plain text directly
        ad_copy_gen.status = 'done'
        ad_copy_gen.updated_at = datetime.utcnow()
    else:
        ad_copy_gen = TxtAdCopyGeneration(
            job_id=job_id, llm_trace_id=llm_trace.llm_trace_id,
            generation_stage='ad_copy_eng', ad_copy_eng=ad_copy_eng_text, status='done' # FIX: Save plain text directly
        )
        db.add(ad_copy_gen)
        
    # Update job status
    job = await db.get(Job, job_id)
    if job:
        job.current_step = 'ad_copy_eng'
        job.status = 'done'
        job.updated_at = datetime.utcnow()
        
    await db.commit()
    
    return {"job_id": str(job_id), "status": "done"}

@app.post("/api/js/gpt/ad-copy-kor")
async def gpt_ad_copy_kor(req: AdCopyKorRequest, db: AsyncSession = Depends(get_db)):
    """(FIXED) Creates a single Korean ad copy and saves it to txt_ad_copy_generations."""
    job_id = uuid.UUID(req.job_id)
    model_name = "gpt-4o-mini"
    llm_model = await get_or_create_llm_model(db, model_name)
    
    eng_gen_res = await db.execute(select(TxtAdCopyGeneration).filter_by(job_id=job_id, generation_stage='ad_copy_eng'))
    eng_gen = eng_gen_res.scalar_one_or_none()
    if not eng_gen or not eng_gen.ad_copy_eng:
        raise HTTPException(status_code=404, detail="English ad copy not found. Run ad-copy-eng first.")
    
    # FIX: Call GPT expecting plain text (expect_json=False)
    ad_copy_kor_text, metadata = await call_gpt(ENG_TO_KOR_TRANSLATION_PROMPT, eng_gen.ad_copy_eng, model_name=model_name, expect_json=False)
    
    # Create LLM trace
    llm_trace = LLMTrace(
        job_id = job_id,
        provider = 'openai', # Corrected provider name
        llm_model_id = llm_model.llm_model_id, # FIX: Use the ID from the fetched model object
        operation_type = 'ad_copy_kor',
        request = metadata['request'],
        response = {"content": metadata['response']},
        latency_ms = metadata['latency_ms'],
        prompt_tokens = metadata['token_usage'].get('prompt_tokens'),
        completion_tokens = metadata['token_usage'].get('completion_tokens'),
        total_tokens = metadata['token_usage'].get('total_tokens'),
        token_usage = metadata['token_usage']
    )
    db.add(llm_trace)
    await db.flush()
    
    # UPSERT logic for txt_ad_copy_generations
    gen_res = await db.execute(select(TxtAdCopyGeneration).filter_by(job_id=job_id, generation_stage='ad_copy_kor'))
    ko_record = gen_res.scalar_one_or_none()

    if ko_record:
        ko_record.llm_trace_id = llm_trace.llm_trace_id
        ko_record.ad_copy_kor = ad_copy_kor_text # FIX: Save plain text directly
        ko_record.status = 'done'
        ko_record.updated_at = datetime.utcnow()
    else:
        ko_record = TxtAdCopyGeneration(
            job_id=job_id, llm_trace_id=llm_trace.llm_trace_id,
            generation_stage='ad_copy_kor', ad_copy_kor=ad_copy_kor_text, status='done' # FIX: Save plain text directly
        )
        db.add(ko_record)
        
    # Update job status (final step)
    job = await db.get(Job, job_id)
    if job:
        # FIX: Set current_step to 'user_img_input' as requested at the end of the pipeline.
        job.current_step = 'user_img_input' 
        job.status = 'done'
        job.updated_at = datetime.utcnow()
        
    await db.commit()
    
    return {"job_id": str(job_id), "status": "done"}

# --- CORRECTED /results endpoint ---
@app.get("/api/v1/jobs/{job_id}/results", status_code=200)
async def get_job_results(job_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """
    job_id에 대한 최종 결과물(이미지 URL 목록, 광고 문구)을 반환합니다.
    Job이 아직 처리 중인 경우 202 상태 코드를 반환하여 폴링을 계속하도록 합니다.
    """
    # 1. Check if job exists and has correct status
    job_result = await db.execute(select(Job).where(Job.job_id == job_id))
    job = job_result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    
    # 2. Check if job is ready (current_step = 'instagram_feed_gen' and status = 'done')
    if job.current_step != 'instagram_feed_gen' or job.status != 'done':
        raise HTTPException(
            status_code=202,  # Accepted - job is still processing
            detail=f"Job is not ready yet. Current step: {job.current_step}, Status: {job.status}"
        )
    
    # 3. Fetch 3 images from JobVariant (overlaid images)
    image_stmt = (
        select(ImageAsset.image_url)
        .join(JobVariant, ImageAsset.image_asset_id == JobVariant.overlaid_img_asset_id)
        .where(JobVariant.job_id == job_id)
        .where(JobVariant.overlaid_img_asset_id.isnot(None))
        .order_by(JobVariant.creation_order)
        .limit(3)
    )
    image_result = await db.execute(image_stmt)
    image_paths = image_result.scalars().all()

    # 4. Check if we have exactly 3 images
    if len(image_paths) < 3:
        raise HTTPException(
            status_code=202,  # Accepted - images are still being generated
            detail=f"Images are still being generated. Found {len(image_paths)}/3 images."
        )
    
    # 5. Fetch Instagram feed data
    feed_stmt = select(InstagramFeed).where(InstagramFeed.job_id == job_id).order_by(InstagramFeed.created_at.desc())
    feed_result = await db.execute(feed_stmt)
    feed_data = feed_result.scalars().first()

    if not feed_data:
        raise HTTPException(
            status_code=202,  # Accepted - feed is still being generated
            detail="Instagram feed is still being generated."
        )

    if not feed_data.instagram_ad_copy or not feed_data.hashtags:
        raise HTTPException(
            status_code=202,  # Accepted - feed content is incomplete
            detail="Instagram feed content is incomplete."
        )

    # 6. Return final results
    # --- FIX: Wrap each image path in an object to match frontend expectation ---
    images_as_objects = [{"image_url": path} for path in image_paths]
    
    return {
        "job_id": job_id,
        "images": images_as_objects,
        "instagram_ad_copy": feed_data.instagram_ad_copy,
        "hashtags": feed_data.hashtags
    }

# ============================
# 🖼️ Image Upload Endpoint
# ============================

# --- FIX: Change the endpoint URL to match the expected API structure ---
@app.post("/api/v1/jobs/create", status_code=200)
async def create_job(
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    # Log raw inputs to confirm parser
    print("=" * 50)
    print("DEBUG create_job called")
    print("DEBUG content-type:", request.headers.get("content-type"))
    
    # Try to parse as multipart form data
    try:
        form_data = await request.form()
        print("DEBUG form_data keys:", list(form_data.keys()))
        print("DEBUG form_data type:", type(form_data))
        
        # React Native FormData sends { uri, name, type } which gets stringified to "[object Object]"
        # We need to iterate through all form entries to find the actual file
        image = None
        request_field = None
        description_field = None
        
        # Iterate through all form entries
        for key in form_data.keys():
            value = form_data.get(key)
            print(f"DEBUG form key '{key}': type={type(value)}, has_read={hasattr(value, 'read') if value else False}")
            
            if key == "image":
                # Check if it's an UploadFile
                if hasattr(value, 'read') and hasattr(value, 'filename'):
                    image = value
                    print(f"DEBUG Found UploadFile: filename={value.filename}")
                else:
                    print(f"WARNING: 'image' field is not an UploadFile: {type(value)}, value={str(value)[:100]}")
            elif key == "image_base64":
                # React Native에서 base64로 인코딩된 이미지
                image_base64 = value
                image_name = form_data.get("image_name", "upload.jpg")
                print(f"DEBUG Found base64 image: name={image_name}, length={len(image_base64) if isinstance(image_base64, str) else 'N/A'}")
            elif key == "image_name":
                # 이미지 이름은 위에서 처리됨
                pass
            elif key == "request":
                request_field = value
            elif key == "description":
                description_field = value
        
        # If image is still None, try to get it from form_data._list (internal structure)
        if image is None:
            print("WARNING: Image not found in form_data.keys(), trying _list")
            if hasattr(form_data, '_list'):
                print(f"DEBUG _list type: {type(form_data._list)}, length: {len(form_data._list) if hasattr(form_data._list, '__len__') else 'N/A'}")
                for idx, item in enumerate(form_data._list):
                    print(f"DEBUG _list[{idx}]: type={type(item)}, value={str(item)[:100] if not hasattr(item, 'read') else 'UploadFile'}")
                    if isinstance(item, tuple) and len(item) >= 2:
                        key, value = item[0], item[1]
                        print(f"DEBUG _list item: key={key}, value_type={type(value)}")
                        if key == "image" and hasattr(value, 'read'):
                            image = value
                            print(f"DEBUG Found UploadFile in _list: filename={value.filename}")
                            break
        
        # Last resort: try to parse multipart body manually
        if image is None:
            print("WARNING: Image still not found, trying to parse multipart body manually")
            try:
                # Reset the request body stream
                body = await request.body()
                print(f"DEBUG body length: {len(body)}")
                
                # Try to use python-multipart to parse
                from multipart import parse_form_data
                import io
                
                # Parse the multipart data
                content_type = request.headers.get("content-type", "")
                boundary = content_type.split("boundary=")[-1] if "boundary=" in content_type else None
                
                if boundary:
                    print(f"DEBUG boundary: {boundary}")
                    # Create a file-like object from body
                    body_stream = io.BytesIO(body)
                    
                    # Parse multipart form data
                    fields, files = parse_form_data(body_stream, boundary.encode())
                    print(f"DEBUG parsed fields: {list(fields.keys())}")
                    print(f"DEBUG parsed files: {list(files.keys())}")
                    
                    if "image" in files:
                        file_info = files["image"]
                        print(f"DEBUG file_info: {file_info}")
                        # file_info is a tuple (filename, file_object)
                        if isinstance(file_info, tuple) and len(file_info) >= 2:
                            filename, file_obj = file_info[0], file_info[1]
                            # Read the file content
                            file_content = file_obj.read()
                            # Create a temporary UploadFile-like object
                            from fastapi import UploadFile as FastAPIUploadFile
                            from io import BytesIO
                            
                            image = FastAPIUploadFile(
                                filename=filename or "upload.jpg",
                                file=BytesIO(file_content)
                            )
                            print(f"DEBUG Created UploadFile from manual parsing: filename={filename}")
            except Exception as e:
                print(f"ERROR in manual multipart parsing: {e}")
                import traceback
                traceback.print_exc()
        
        # If image is still None, check if we have base64 image
        if image is None and 'image_base64' in locals() and image_base64:
            print("DEBUG Processing base64 image")
            try:
                import base64
                from io import BytesIO
                from fastapi import UploadFile as FastAPIUploadFile
                
                # Decode base64 image
                image_data = base64.b64decode(image_base64)
                image = FastAPIUploadFile(
                    filename=image_name if 'image_name' in locals() else "upload.jpg",
                    file=BytesIO(image_data)
                )
                print(f"DEBUG Created UploadFile from base64: filename={image.filename}")
            except Exception as e:
                print(f"ERROR decoding base64 image: {e}")
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=422, detail=f"Failed to decode base64 image: {str(e)}")
        
        if image is None:
            raise HTTPException(
                status_code=422,
                detail="Could not extract image file from FormData. React Native FormData may not be properly configured."
            )
        
        print("DEBUG image filename:", image.filename if hasattr(image, 'filename') else 'N/A')
        print("DEBUG image content_type:", image.content_type if hasattr(image, 'content_type') else 'N/A')
        
        # Get request or description field
        request_field = form_data.get("request")
        description_field = form_data.get("description")
        
        print("DEBUG request field:", request_field)
        print("DEBUG description field:", description_field)
        print("=" * 50)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR parsing form data: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=422, detail=f"Failed to parse form data: {str(e)}")
    
    # Prefer 'request' JSON, fallback to plain 'description'
    raw = request_field or description_field
    if not raw:
        raise HTTPException(status_code=422, detail="Missing form field: 'request' or 'description'")

    if not isinstance(raw, str):
        raw = str(raw)

    try:
        data = json.loads(raw) if raw.strip().startswith("{") else {"description": raw}
        description_text = (data.get("description") or raw).strip()
        # Pull optional tenant/store from payload (no defaults)
        tenant_id = data.get("tenant_id")
        store_id_uuid = None
        store_id_raw = data.get("store_id")
        if store_id_raw:
            try:
                store_id_uuid = uuid.UUID(str(store_id_raw))
            except Exception:
                store_id_uuid = None
        if not description_text:
            raise ValueError("Description is empty")
    except json.JSONDecodeError:
        description_text = raw.strip()
        tenant_id = None
        store_id_uuid = None
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid description payload: {e}")

    # --- FIX: Force a fixed tenant_id for all jobs ---
    tenant_id = "user_0b1bfa70-20a4-4807-a5b1-397e3c197ab8"
    print(f"DEBUG: Forcing fixed tenant_id: {tenant_id}")

    # Save image to the correct asset location
    # Path format: /assets/js/tenants/{tenant_id}/original/{year}/{month:02d}/{day:02d}/{filename}
    image_filename = image.filename if image.filename else 'upload.jpg'
    file_ext = os.path.splitext(image_filename)[1]
    unique_filename = f"{uuid.uuid4().hex}{file_ext}"
    
    # Set kind to "original" for create_job()
    kind = "original"
    
    # Get current date
    now = datetime.utcnow()
    year = now.year
    month = f"{now.month:02d}"
    day = f"{now.day:02d}"
    
    # Full directory path on the server's filesystem
    # e.g., /opt/feedlyai/assets/js/tenants/{tenant_id}/original/2025/12/03
    save_dir = os.path.join(ASSETS_ROOT, "js", "tenants", tenant_id, kind, str(year), month, day)
    os.makedirs(save_dir, exist_ok=True)
    
    # Final, full path to save the file on the server
    save_path = os.path.join(save_dir, unique_filename)
    
    # Save the image file
    try:
        image_content = await image.read()
        with open(save_path, "wb") as f:
            f.write(image_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save image: {e}")
    
    # This is the web-accessible URL path to store in the database
    # e.g., /assets/js/tenants/{tenant_id}/original/2025/12/03/some_uuid.jpg
    path = f"/assets/js/tenants/{tenant_id}/{kind}/{year}/{month}/{day}/{unique_filename}"

    # Reference existing records from tone_styles, users, tenants tables
    # 1. Get or create user based on DEFAULT_USER_ID
    user_obj = None
    creator_id = None
    if DEFAULT_USER_ID:
        try:
            user_uuid = uuid.UUID(str(DEFAULT_USER_ID))
            user_result = await db.execute(select(User).filter(User.user_id == user_uuid))
            user_obj = user_result.scalar_one_or_none()
            if user_obj:
                creator_id = user_obj.user_id
                print(f"DEBUG: Found user: {creator_id}")
            else:
                # User doesn't exist, create a new one
                print(f"INFO: User '{DEFAULT_USER_ID}' not found, creating new user")
                user_obj = User(
                    user_id=user_uuid,
                    uid=f"user_{user_uuid}"
                )
                db.add(user_obj)
                await db.flush()
                creator_id = user_obj.user_id
                print(f"DEBUG: Created new user: {creator_id}")
        except Exception as e:
            print(f"ERROR: Invalid DEFAULT_USER_ID '{DEFAULT_USER_ID}': {e}")
            raise HTTPException(status_code=400, detail=f"Invalid DEFAULT_USER_ID: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="DEFAULT_USER_ID is required but not set in environment variables")
    
    # 2. Get or create tenant based on user (user-specific tenant)
    tenant_obj = None
    if tenant_id:
        # If tenant_id is provided in request, use it
        tenant_result = await db.execute(select(Tenant).filter(Tenant.tenant_id == tenant_id))
        tenant_obj = tenant_result.scalar_one_or_none()
        if not tenant_obj:
            # Tenant doesn't exist, create a new one
            print(f"INFO: Tenant '{tenant_id}' not found in database, creating new tenant")
            tenant_obj = Tenant(
                tenant_id=tenant_id,
                display_name=f"Tenant {tenant_id}",
                uid=f"tenant_{tenant_id}"
            )
            db.add(tenant_obj)
            await db.flush()
            print(f"DEBUG: Created new tenant: {tenant_id}")
    # --- REMOVED: This 'else' block is no longer reachable because tenant_id is always set ---
    # else:
    #     # No tenant_id provided, create user-specific tenant
    #     user_tenant_id = f"user_{creator_id}"
    #     tenant_result = await db.execute(select(Tenant).filter(Tenant.tenant_id == user_tenant_id))
    #     tenant_obj = tenant_result.scalar_one_or_none()
    #     if tenant_obj:
    #         tenant_id = user_tenant_id
    #         print(f"DEBUG: Using existing user tenant: {tenant_id}")
    #     else:
    #         # Create user-specific tenant
    #         print(f"INFO: Creating user-specific tenant: {user_tenant_id}")
    #         tenant_obj = Tenant(
    #             tenant_id=user_tenant_id,
    #             display_name=f"User Tenant {creator_id}",
    #             uid=f"tenant_{user_tenant_id}"
    #         )
    #         db.add(tenant_obj)
    #         await db.flush()
    #         tenant_id = user_tenant_id
    #         print(f"DEBUG: Created user-specific tenant: {tenant_id}")
    
    # 3. Get or create store for the user
    store_obj = None
    # Check if store_id was provided in request
    if store_id_uuid:
        # If store_id is provided in request, verify it exists
        store_result = await db.execute(select(Store).filter(Store.store_id == store_id_uuid))
        store_obj = store_result.scalar_one_or_none()
        if not store_obj:
            print(f"WARNING: Store '{store_id_uuid}' not found in database, will create new one")
            store_id_uuid = None
    
    if not store_id_uuid:
        # No store_id provided or not found, find or create user-specific store
        store_result = await db.execute(select(Store).filter(Store.user_id == creator_id).limit(1))
        store_obj = store_result.scalar_one_or_none()
        if store_obj:
            store_id_uuid = store_obj.store_id
            print(f"DEBUG: Using existing user store: {store_id_uuid}")
        else:
            # Create user-specific store
            print(f"INFO: Creating user-specific store for user: {creator_id}")
            store_obj = Store(
                user_id=creator_id,
                title=f"Store for User {creator_id}",
                body=f"Default store for user {creator_id}",
                store_category="default"
            )
            db.add(store_obj)
            await db.flush()
            store_id_uuid = store_obj.store_id
            print(f"DEBUG: Created user-specific store: {store_id_uuid}")
    
    # 4. Get default tone_style (optional, can be None)
    tone_style_id = None
    try:
        # Try to get the first active tone_style (or any tone_style if no active flag)
        tone_style_result = await db.execute(select(ToneStyle).limit(1))
        tone_style_obj = tone_style_result.scalar_one_or_none()
        if tone_style_obj:
            tone_style_id = tone_style_obj.tone_style_id
            print(f"DEBUG: Using tone_style_id: {tone_style_id}")
        else:
            print("WARNING: No tone_style found in database, using None")
    except Exception as e:
        print(f"WARNING: Error fetching tone_style: {e}, using None")

    # DB writes (ensure column names match schema)
    new_job = Job(
        # The job is first created with this state
        status='done',
        current_step='job_created',
        tenant_id=tenant_id,
        store_id=store_id_uuid  # Use the store_id we found or created
    )
    db.add(new_job)
    await db.flush() # Flush to get the new_job.job_id

    new_image = ImageAsset(
        image_url=path,
        tenant_id=tenant_id,
        creator_id=creator_id,
        job_id=new_job.job_id  # FIX: Assign the job_id to the image asset
    )

    db.add(new_image)
    await db.flush() # Flush to get the new_image.image_asset_id

    new_input = JobInput(
        job_id=new_job.job_id,
        img_asset_id=new_image.image_asset_id,
        tone_style_id=tone_style_id,
        desc_kor=description_text  # use parsed description
    )
    db.add(new_input)

    # Immediately update the job to its next state before committing.
    # This is the final signal to the listener.
    new_job.current_step = 'user_img_input'
    new_job.status = 'done' 
    new_job.updated_at = datetime.utcnow()
    
    await db.commit()
    return {"job_id": str(new_job.job_id), "status": "job_created_and_inputs_submitted"}

@app.post("/api/js/gpt/kor-to-eng")
async def gpt_kor_to_eng(req: KorToEngRequest, db: AsyncSession = Depends(get_db)):
    """(FIXED) Translates Korean description and updates job_inputs.desc_eng."""
    try:
        job_id = uuid.UUID(req.job_id)
        model_name = "gpt-4o-mini"
        llm_model = await get_or_create_llm_model(db, model_name)
        
        job_input_res = await db.execute(select(JobInput).filter(JobInput.job_id == job_id))
        job_input = job_input_res.scalar_one_or_none()
        if not job_input or not job_input.desc_kor:
            raise HTTPException(status_code=404, detail="Job input with Korean description not found.")
        
        # This step still expects JSON, so expect_json=True
        gpt_result, metadata = await call_gpt(TRANSLATION_SYSTEM_PROMPT, job_input.desc_kor, model_name=model_name, expect_json=True)
        desc_eng = gpt_result.get("translation_en", "Translation failed.")
        
        llm_trace = LLMTrace(
            job_id = job_id,
            provider = 'openai', # Corrected provider name
            llm_model_id = llm_model.llm_model_id, # FIX: Use the ID from the fetched model object
            operation_type = 'kor_to_eng',
            request = {"system": TRANSLATION_SYSTEM_PROMPT, "user": job_input.desc_kor},
            response = {"content": desc_eng},
            latency_ms = metadata.get('latency_ms', 0),
            prompt_tokens = metadata.get('token_usage', {}).get('prompt_tokens'),
            completion_tokens = metadata.get('token_usage', {}).get('completion_tokens'),
            total_tokens = metadata.get('token_usage', {}).get('total_tokens'),
            token_usage = metadata.get('token_usage', {})
        )
        db.add(llm_trace)
        
        # 4. UPDATE the desc_eng field in the job_inputs table
        job_input.desc_eng = desc_eng
        job_input.updated_at = datetime.utcnow()
        
        # 5. Update job status to signal completion of this step
        job = await db.get(Job, job_id)
        if job:
            job.current_step = 'kor_to_eng'
            job.status = 'done'
            job.updated_at = datetime.utcnow()
        
        await db.commit()
        
        return {"job_id": str(job_id), "status": "done", "desc_eng": desc_eng}
    except Exception as e:
        print(f"ERROR in kor-to-eng: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/js/gpt/ad-copy-eng")
async def gpt_ad_copy_eng(req: AdCopyEngRequest, db: AsyncSession = Depends(get_db)):
    """(FIXED) Creates a single English ad copy and saves it to txt_ad_copy_generations."""
    job_id = uuid.UUID(req.job_id)
    model_name = "gpt-4o-mini"
    llm_model = await get_or_create_llm_model(db, model_name)
    
    job_input_res = await db.execute(select(JobInput).filter(JobInput.job_id == job_id))
    job_input = job_input_res.scalar_one_or_none()
    if not job_input or not job_input.desc_eng:
        raise HTTPException(status_code=404, detail="English description not found in job_inputs. Run kor-to-eng first.")
    
    # FIX: Call GPT expecting plain text (expect_json=False)
    user_prompt = f"Description (English): {job_input.desc_eng}"
    ad_copy_eng_text, metadata = await call_gpt(GPT_COPY_SYSTEM_PROMPT, user_prompt, model_name=model_name, expect_json=False)
    
    # Create LLM trace
    llm_trace = LLMTrace(
        job_id = job_id,
        provider = 'openai', # Corrected provider name
        llm_model_id = llm_model.llm_model_id, # FIX: Use the ID from the fetched model object
        operation_type = 'ad_copy_gen',
        request = metadata['request'],
        response = {"content": metadata['response']},
        latency_ms = metadata['latency_ms'],
        prompt_tokens = metadata['token_usage'].get('prompt_tokens'),
        completion_tokens = metadata['token_usage'].get('completion_tokens'),
        total_tokens = metadata['token_usage'].get('total_tokens'),
        token_usage = metadata['token_usage']
    )
    db.add(llm_trace)
    await db.flush()
    
    # UPSERT logic for txt_ad_copy_generations
    gen_res = await db.execute(select(TxtAdCopyGeneration).filter_by(job_id=job_id, generation_stage='ad_copy_eng'))
    ad_copy_gen = gen_res.scalar_one_or_none()
    
    if ad_copy_gen:
        ad_copy_gen.llm_trace_id = llm_trace.llm_trace_id
        ad_copy_gen.ad_copy_eng = ad_copy_eng_text # FIX: Save plain text directly
        ad_copy_gen.status = 'done'
        ad_copy_gen.updated_at = datetime.utcnow()
    else:
        ad_copy_gen = TxtAdCopyGeneration(
            job_id=job_id, llm_trace_id=llm_trace.llm_trace_id,
            generation_stage='ad_copy_eng', ad_copy_eng=ad_copy_eng_text, status='done' # FIX: Save plain text directly
        )
        db.add(ad_copy_gen)
        
    # Update job status
    job = await db.get(Job, job_id)
    if job:
        job.current_step = 'ad_copy_eng'
        job.status = 'done'
        job.updated_at = datetime.utcnow()
        
    await db.commit()
    
    return {"job_id": str(job_id), "status": "done"}

@app.post("/api/js/gpt/ad-copy-kor")
async def gpt_ad_copy_kor(req: AdCopyKorRequest, db: AsyncSession = Depends(get_db)):
    """(FIXED) Creates a single Korean ad copy and saves it to txt_ad_copy_generations."""
    job_id = uuid.UUID(req.job_id)
    model_name = "gpt-4o-mini"
    llm_model = await get_or_create_llm_model(db, model_name)
    
    eng_gen_res = await db.execute(select(TxtAdCopyGeneration).filter_by(job_id=job_id, generation_stage='ad_copy_eng'))
    eng_gen = eng_gen_res.scalar_one_or_none()
    if not eng_gen or not eng_gen.ad_copy_eng:
        raise HTTPException(status_code=404, detail="English ad copy not found. Run ad-copy-eng first.")
    
    # FIX: Call GPT expecting plain text (expect_json=False)
    ad_copy_kor_text, metadata = await call_gpt(ENG_TO_KOR_TRANSLATION_PROMPT, eng_gen.ad_copy_eng, model_name=model_name, expect_json=False)
    
    # Create LLM trace
    llm_trace = LLMTrace(
        job_id = job_id,
        provider = 'openai', # Corrected provider name
        llm_model_id = llm_model.llm_model_id, # FIX: Use the ID from the fetched model object
        operation_type = 'ad_copy_kor',
        request = metadata['request'],
        response = {"content": metadata['response']},
        latency_ms = metadata['latency_ms'],
        prompt_tokens = metadata['token_usage'].get('prompt_tokens'),
        completion_tokens = metadata['token_usage'].get('completion_tokens'),
        total_tokens = metadata['token_usage'].get('total_tokens'),
        token_usage = metadata['token_usage']
    )
    db.add(llm_trace)
    await db.flush()
    
    # UPSERT logic for txt_ad_copy_generations
    gen_res = await db.execute(select(TxtAdCopyGeneration).filter_by(job_id=job_id, generation_stage='ad_copy_kor'))
    ko_record = gen_res.scalar_one_or_none()

    if ko_record:
        ko_record.llm_trace_id = llm_trace.llm_trace_id
        ko_record.ad_copy_kor = ad_copy_kor_text # FIX: Save plain text directly
        ko_record.status = 'done'
        ko_record.updated_at = datetime.utcnow()
    else:
        ko_record = TxtAdCopyGeneration(
            job_id=job_id, llm_trace_id=llm_trace.llm_trace_id,
            generation_stage='ad_copy_kor', ad_copy_kor=ad_copy_kor_text, status='done' # FIX: Save plain text directly
        )
        db.add(ko_record)
        
    # Update job status (final step)
    job = await db.get(Job, job_id)
    if job:
        # FIX: Set current_step to 'user_img_input' as requested at the end of the pipeline.
        job.current_step = 'user_img_input' 
        job.status = 'done'
        job.updated_at = datetime.utcnow()
        
    await db.commit()
    
    return {"job_id": str(job_id), "status": "done"}

# --- CORRECTED /results endpoint ---
@app.get("/api/v1/jobs/{job_id}/results", status_code=200)
async def get_job_results(job_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """
    job_id에 대한 최종 결과물(이미지 URL 목록, 광고 문구)을 반환합니다.
    Job이 아직 처리 중인 경우 202 상태 코드를 반환하여 폴링을 계속하도록 합니다.
    """
    # 1. Check if job exists and has correct status
    job_result = await db.execute(select(Job).where(Job.job_id == job_id))
    job = job_result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    
    # 2. Check if job is ready (current_step = 'instagram_feed_gen' and status = 'done')
    if job.current_step != 'instagram_feed_gen' or job.status != 'done':
        raise HTTPException(
            status_code=202,  # Accepted - job is still processing
            detail=f"Job is not ready yet. Current step: {job.current_step}, Status: {job.status}"
        )
    
    # 3. Fetch 3 images from JobVariant (overlaid images)
    image_stmt = (
        select(ImageAsset.image_url)
        .join(JobVariant, ImageAsset.image_asset_id == JobVariant.overlaid_img_asset_id)
        .where(JobVariant.job_id == job_id)
        .where(JobVariant.overlaid_img_asset_id.isnot(None))
        .order_by(JobVariant.creation_order)
        .limit(3)
    )
    image_result = await db.execute(image_stmt)
    image_paths = image_result.scalars().all()

    # 4. Check if we have exactly 3 images
    if len(image_paths) < 3:
        raise HTTPException(
            status_code=202,  # Accepted - images are still being generated
            detail=f"Images are still being generated. Found {len(image_paths)}/3 images."
        )
    
    # 5. Fetch Instagram feed data
    feed_stmt = select(InstagramFeed).where(InstagramFeed.job_id == job_id).order_by(InstagramFeed.created_at.desc())
    feed_result = await db.execute(feed_stmt)
    feed_data = feed_result.scalars().first()

    if not feed_data:
        raise HTTPException(
            status_code=202,  # Accepted - feed is still being generated
            detail="Instagram feed is still being generated."
        )

    if not feed_data.instagram_ad_copy or not feed_data.hashtags:
        raise HTTPException(
            status_code=202,  # Accepted - feed content is incomplete
            detail="Instagram feed content is incomplete."
        )

    # 6. Return final results
    # --- FIX: Wrap each image path in an object to match frontend expectation ---
    images_as_objects = [{"image_url": path} for path in image_paths]
    
    return {
        "job_id": job_id,
        "images": images_as_objects,
        "instagram_ad_copy": feed_data.instagram_ad_copy,
        "hashtags": feed_data.hashtags
    }