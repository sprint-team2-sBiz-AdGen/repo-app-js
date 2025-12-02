# updated at: 2025-12-02
# version: 1.6.0

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
from pydantic import BaseModel
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

async def call_gpt(system_prompt: str, user_prompt: str, model_name: str = "gpt-4o-mini") -> Tuple[dict, dict]:
    """Call GPT API and return (result, metadata) where metadata includes latency and token usage"""
    import time
    start_time = time.time()
    try:
        resp = await asyncio.to_thread(
            client.chat.completions.create,
            model=model_name,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.7,
        )
        latency_ms = (time.time() - start_time) * 1000
        
        # Extract token usage
        token_usage = {}
        if hasattr(resp, 'usage'):
            token_usage = {
                "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
                "total_tokens": resp.usage.total_tokens if resp.usage else 0
            }
        
        result = json.loads(resp.choices[0].message.content)
        metadata = {
            "latency_ms": latency_ms,
            "token_usage": token_usage,
            "model": model_name,
            "request": {"system": system_prompt, "user": user_prompt},
            "response": resp.choices[0].message.content
        }
        return result, metadata
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
    request_obj: StarletteRequest,
    db: AsyncSession = Depends(get_db)
):
    # Log raw inputs to confirm parser
    print("=" * 50)
    print("DEBUG create_job called")
    print("DEBUG content-type:", request_obj.headers.get("content-type"))
    
    # Try to parse as multipart form data
    try:
        form_data = await request_obj.form()
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
                body = await request_obj.body()
                print(f"DEBUG body length: {len(body)}")
                
                # Try to use python-multipart to parse
                from multipart import parse_form_data
                import io
                
                # Parse the multipart data
                content_type = request_obj.headers.get("content-type", "")
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

    # Save image
    image_filename = image.filename if image.filename else 'upload.jpg'
    filename = f"{uuid.uuid4().hex}{os.path.splitext(image_filename)[1]}"
    path = os.path.join(MEDIA_ROOT, filename)
    try:
        image_content = await image.read()
        with open(path, "wb") as f:
            f.write(image_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save image: {e}")

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
    else:
        # No tenant_id provided, create user-specific tenant
        user_tenant_id = f"user_{creator_id}"
        tenant_result = await db.execute(select(Tenant).filter(Tenant.tenant_id == user_tenant_id))
        tenant_obj = tenant_result.scalar_one_or_none()
        if tenant_obj:
            tenant_id = user_tenant_id
            print(f"DEBUG: Using existing user tenant: {tenant_id}")
        else:
            # Create user-specific tenant
            print(f"INFO: Creating user-specific tenant: {user_tenant_id}")
            tenant_obj = Tenant(
                tenant_id=user_tenant_id,
                display_name=f"User Tenant {creator_id}",
                uid=f"tenant_{user_tenant_id}"
            )
            db.add(tenant_obj)
            await db.flush()
            tenant_id = user_tenant_id
            print(f"DEBUG: Created user-specific tenant: {tenant_id}")
    
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
    new_image = ImageAsset(
        image_url=path,
        tenant_id=tenant_id,
        creator_id=creator_id
    )
    new_job = Job(
        status='running',
        current_step='job_created',
        tenant_id=tenant_id,
        store_id=store_id_uuid  # Use the store_id we found or created
    )

    db.add_all([new_image, new_job])
    await db.flush()

    new_input = JobInput(
        job_id=new_job.job_id,
        img_asset_id=new_image.image_asset_id,
        tone_style_id=tone_style_id,
        desc_kor=description_text  # use parsed description
    )
    db.add(new_input)
    await db.commit()
    return {"job_id": str(new_job.job_id), "status": "job_created"}

@app.post("/api/js/gpt/kor-to-eng")
async def gpt_kor_to_eng(req: KorToEngRequest, db: AsyncSession = Depends(get_db)):
    """한국어 설명을 영어로 변환"""
    try:
        job_id = uuid.UUID(req.job_id)
        
        # 1. job_inputs에서 desc_kor 조회
        job_input = (await db.execute(select(JobInput).filter(JobInput.job_id == job_id))).scalar_one_or_none()
        if not job_input:
            raise HTTPException(status_code=404, detail="Job input not found")
        if not job_input.desc_kor:
            raise HTTPException(status_code=400, detail="Korean description is empty")
        
        # 2. GPT API 호출
        gpt_result, metadata = await call_gpt(TRANSLATION_SYSTEM_PROMPT, job_input.desc_kor)
        desc_eng = gpt_result.get("translation_en", "Translation failed.")
        
        # 3. llm_models에서 사용한 모델 조회 (선택적)
        llm_model_id = None
        try:
            llm_model_result = await db.execute(
                select(LLMModel)
                .filter(LLMModel.provider == 'openai', LLMModel.model_name == metadata.get('model', 'gpt-4o-mini'), LLMModel.is_active == 'true')
                .limit(1)
            )
            llm_model = llm_model_result.scalar_one_or_none()
            if llm_model:
                llm_model_id = llm_model.llm_model_id
        except Exception as e:
            print(f"Warning: Could not find LLM model: {e}")
        
        # 4. llm_traces에 기록
        llm_trace = LLMTrace(
            job_id=job_id,
            provider='gpt',
            llm_model_id=llm_model_id,
            operation_type='kor_to_eng',
            request=metadata.get('request', {}),
            response={"content": metadata.get('response', '')},
            latency_ms=metadata.get('latency_ms', 0),
            prompt_tokens=metadata.get('token_usage', {}).get('prompt_tokens'),
            completion_tokens=metadata.get('token_usage', {}).get('completion_tokens'),
            total_tokens=metadata.get('token_usage', {}).get('total_tokens'),
            token_usage=metadata.get('token_usage', {})
        )
        db.add(llm_trace)
        await db.flush()
        
        # 5. txt_ad_copy_generations에 레코드 생성
        ad_copy_gen = TxtAdCopyGeneration(
            job_id=job_id,
            llm_trace_id=llm_trace.llm_trace_id,
            generation_stage='kor_to_eng',
            ad_copy_eng=desc_eng,
            status='done'
        )
        db.add(ad_copy_gen)
        
        # 6. job_inputs.desc_eng 업데이트
        job_input.desc_eng = desc_eng
        
        # 7. jobs 테이블 업데이트
        job = await db.get(Job, job_id)
        if job:
            job.current_step = 'desc_kor_translate'
            job.status = 'done'
            job.updated_at = datetime.utcnow()
        
        ad_copy_gen.updated_at = datetime.utcnow()
        await db.commit()
        
        return {
            "job_id": str(job_id),
            "llm_trace_id": str(llm_trace.llm_trace_id),
            "ad_copy_gen_id": str(ad_copy_gen.ad_copy_gen_id),
            "desc_eng": desc_eng,
            "status": "done"
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR in kor-to-eng: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/js/gpt/ad-copy-eng")
async def gpt_ad_copy_eng(req: AdCopyEngRequest, db: AsyncSession = Depends(get_db)):
    """영어 광고문구 생성"""
    job_id = uuid.UUID(req.job_id)
    
    # 1. job_inputs에서 desc_eng, tone_style_id 조회
    job_input = (await db.execute(select(JobInput).filter(JobInput.job_id == job_id))).scalar_one_or_none()
    if not job_input:
        raise HTTPException(status_code=404, detail="Job input not found")
    if not job_input.desc_eng:
        raise HTTPException(status_code=404, detail="English description not found. Run kor-to-eng first.")
    
    # 2. tone_styles 조회 (선택적, 필요시 추가)
    # tone_style_id가 있다면 조회 가능
    
    # 3. GPT API 호출: 영어 광고문구 생성
    user_prompt = f"Description (Korean): {job_input.desc_kor}\nDescription (English): {job_input.desc_eng}"
    gpt_result, metadata = await call_gpt(GPT_COPY_SYSTEM_PROMPT, user_prompt)
    ad_copy_eng_text = json.dumps(gpt_result.get("variants", []))  # JSON 문자열로 저장
    
    # 4. llm_models에서 사용한 모델 조회 (선택적)
    llm_model = await db.execute(
        select(LLMModel)
        .filter(LLMModel.provider == 'openai', LLMModel.model_name == metadata['model'], LLMModel.is_active == 'true')
        .limit(1)
    )
    llm_model_id = llm_model.scalar_one_or_none()
    
    # 5. llm_traces에 기록
    llm_trace = LLMTrace(
        job_id=job_id,
        provider='gpt',
        llm_model_id=llm_model_id.llm_model_id if llm_model_id else None,
        operation_type='ad_copy_gen',
        request=metadata['request'],
        response={"content": metadata['response']},
        latency_ms=metadata['latency_ms'],
        prompt_tokens=metadata['token_usage'].get('prompt_tokens'),
        completion_tokens=metadata['token_usage'].get('completion_tokens'),
        total_tokens=metadata['token_usage'].get('total_tokens'),
        token_usage=metadata['token_usage']
    )
    db.add(llm_trace)
    await db.flush()
    
    # 6. txt_ad_copy_generations에 레코드 생성/업데이트
    ad_copy_gen = TxtAdCopyGeneration(
        job_id=job_id,
        llm_trace_id=llm_trace.llm_trace_id,
        generation_stage='ad_copy_eng',
        ad_copy_eng=ad_copy_eng_text,
        status='done'
    )
    db.add(ad_copy_gen)
    
    # 7. jobs 테이블 업데이트
    job = await db.get(Job, job_id)
    if job:
        job.current_step = 'ad_copy_gen_eng'
        job.status = 'done'
        job.updated_at = datetime.utcnow()
    
    ad_copy_gen.updated_at = datetime.utcnow()
    await db.commit()
    
    return {
        "job_id": str(job_id),
        "llm_trace_id": str(llm_trace.llm_trace_id),
        "ad_copy_gen_id": str(ad_copy_gen.ad_copy_gen_id),
        "ad_copy_eng": ad_copy_eng_text,
        "status": "done"
    }

@app.post("/api/js/gpt/ad-copy-kor")
async def gpt_ad_copy_kor(req: AdCopyKorRequest, db: AsyncSession = Depends(get_db)):
    """한글 광고문구 생성 (오버레이에 사용)"""
    job_id = uuid.UUID(req.job_id)
    
    # 1. txt_ad_copy_generations에서 ad_copy_eng 조회 (generation_stage='ad_copy_eng')
    eng_gen_result = await db.execute(
        select(TxtAdCopyGeneration)
        .filter(
            TxtAdCopyGeneration.job_id == job_id,
            TxtAdCopyGeneration.generation_stage == 'ad_copy_eng'
        )
        .order_by(TxtAdCopyGeneration.created_at.desc())
    )
    eng_gen = eng_gen_result.scalars().first()
    if not eng_gen or not eng_gen.ad_copy_eng:
        raise HTTPException(status_code=404, detail="English ad copy not found. Run ad-copy-eng first.")
    
    # 2. GPT API 호출: 영어 광고문구 → 한글 광고문구 변환
    gpt_result, metadata = await call_gpt(ENG_TO_KOR_TRANSLATION_PROMPT, eng_gen.ad_copy_eng)
    variants_ko = gpt_result.get("variants", [])
    if not variants_ko:
        raise HTTPException(status_code=500, detail="Translation failed.")
    
    # 한글 광고문구를 텍스트로 변환 (첫 번째 variant의 copy_ko 사용 또는 전체 JSON)
    ad_copy_kor_text = json.dumps(variants_ko) if isinstance(variants_ko, list) else str(variants_ko)
    
    # 3. llm_models에서 사용한 모델 조회 (선택적)
    llm_model = await db.execute(
        select(LLMModel)
        .filter(LLMModel.provider == 'openai', LLMModel.model_name == metadata['model'], LLMModel.is_active == 'true')
        .limit(1)
    )
    llm_model_id = llm_model.scalar_one_or_none()
    
    # 4. llm_traces에 기록
    llm_trace = LLMTrace(
        job_id=job_id,
        provider='gpt',
        llm_model_id=llm_model_id.llm_model_id if llm_model_id else None,
        operation_type='ad_copy_kor',
        request=metadata['request'],
        response={"content": metadata['response']},
        latency_ms=metadata['latency_ms'],
        prompt_tokens=metadata['token_usage'].get('prompt_tokens'),
        completion_tokens=metadata['token_usage'].get('completion_tokens'),
        total_tokens=metadata['token_usage'].get('total_tokens'),
        token_usage=metadata['token_usage']
    )
    db.add(llm_trace)
    await db.flush()
    
    # 5. txt_ad_copy_generations에 레코드 생성
    ko_record = TxtAdCopyGeneration(
        job_id=job_id,
        llm_trace_id=llm_trace.llm_trace_id,
        generation_stage='ad_copy_kor',
        ad_copy_kor=ad_copy_kor_text,
        status='done'
    )
    db.add(ko_record)
    
    # 6. jobs 테이블 업데이트
    job = await db.get(Job, job_id)
    if job:
        job.current_step = 'ad_copy_gen_kor'
        job.status = 'done'
        job.updated_at = datetime.utcnow()
    
    ko_record.updated_at = datetime.utcnow()
    await db.commit()
    
    return {
        "job_id": str(job_id),
        "llm_trace_id": str(llm_trace.llm_trace_id),
        "ad_copy_gen_id": str(ko_record.ad_copy_gen_id),
        "ad_copy_kor": ad_copy_kor_text,
        "status": "done"
    }
