# JS 파트 구현 가이드

## 📋 개요

이 문서는 **JS 파트**에서 구현해야 할 광고문구 생성 관련 기능에 대한 가이드입니다.

**작성일**: 2025-12-01  
**최종 수정일**: 2025-12-01  
**버전**: 1.2.0  
**작성자**: LEEYH205

---

## 🎯 JS 파트 담당 범위

JS 파트는 다음 세 단계를 담당합니다:

1. **`kor_to_eng`**: 한국어 설명 → 영어 변환
2. **`ad_copy_eng`**: 영어 광고문구 생성
3. **`ad_copy_kor`**: 한글 광고문구 생성 (오버레이에 사용)

---

## 📊 데이터베이스 구조

### 1. 관련 테이블

#### `job_inputs` 테이블
- `job_id` (UUID, PK): Job ID
- `desc_kor` (TEXT): 사용자 입력 - 한국어 설명 (30자 이내)
- `desc_eng` (TEXT): GPT Kor→Eng 변환 결과 또는 영어 광고문구
- `tone_style_id` (UUID, FK): 톤 & 스타일 ID

#### `jobs` 테이블
- `job_id` (UUID, PK): Job ID
- `store_id` (UUID, FK → stores): 스토어 ID

#### `stores` 테이블
- `store_id` (UUID, PK): 스토어 ID
- `user_id` (UUID, FK → users): 사용자 ID
- `image_id` (UUID, FK → image_assets): 이미지 ID
- `title` (VARCHAR): 스토어 제목
- `body` (TEXT): 스토어 설명
- `store_category` (TEXT): 스토어 카테고리
- `auto_scoring_flag` (BOOLEAN): 자동 점수 계산 플래그

**스토어 정보 조회 방법:**
```sql
SELECT s.title, s.body, s.store_category
FROM jobs j
INNER JOIN stores s ON j.store_id = s.store_id
WHERE j.job_id = :job_id
```
- **참고**: 스토어 정보는 `jobs.store_id`를 통해 `stores` 테이블에서 조회
- `job_inputs` 테이블에 `store_information` 컬럼 추가 불필요

#### `txt_ad_copy_generations` 테이블 (신규)
- `ad_copy_gen_id` (UUID, PK)
- `job_id` (UUID, FK → jobs): Job과 연결
- `llm_trace_id` (UUID, FK → llm_traces): GPT API 호출 Trace 참조
- `generation_stage` (TEXT): 생성 단계
  - `'kor_to_eng'`: 한국어 → 영어 변환
  - `'ad_copy_eng'`: 영어 광고문구 생성
  - `'ad_copy_kor'`: 한글 광고문구 생성 (오버레이에 사용)
- `ad_copy_eng` (TEXT): 영어 광고문구
- `ad_copy_kor` (TEXT): 한글 광고문구 (오버레이에 사용)
- `status` (TEXT): 'queued', 'running', 'done', 'failed'
- `created_at`, `updated_at`

#### `llm_models` 테이블 (참고)
- `llm_model_id` (UUID, PK): LLM 모델 고유 식별자
- `model_name` (VARCHAR): 모델 이름 (예: 'gpt-4o-mini')
- `model_version` (VARCHAR): 모델 버전 (예: '2024-07-18')
- `provider` (VARCHAR): 제공자 (예: 'openai', 'anthropic', 'google')
- `default_temperature` (FLOAT): 기본 temperature 설정
- `default_max_tokens` (INTEGER): 기본 최대 토큰 수
- `is_active` (VARCHAR): 활성화 여부 ('true', 'false')
- `created_at`, `updated_at`

**LLM 모델 조회 예시:**
```sql
SELECT llm_model_id, model_name, default_temperature, default_max_tokens
FROM llm_models
WHERE provider = 'openai'
  AND model_name = 'gpt-4o-mini'
  AND is_active = 'true'
LIMIT 1
```

#### `llm_traces` 테이블
- `llm_trace_id` (UUID, PK)
- `job_id` (UUID, FK → jobs)
- `provider` (TEXT): 'gpt' 등
- `llm_model_id` (UUID, FK → llm_models): 사용된 LLM 모델 참조 (선택적)
- `tone_style_id` (UUID, FK → tone_styles): 톤 스타일 ID (선택적)
- `enhanced_img_id` (UUID, FK → image_assets): 향상된 이미지 ID (선택적)
- `prompt_id` (UUID): 프롬프트 ID (선택적)
- `operation_type` (TEXT): 'kor_to_eng', 'ad_copy_gen', 'ad_copy_kor' 등
- `request` (JSONB): GPT API 요청 데이터
- `response` (JSONB): GPT API 응답 데이터
- `latency_ms` (FLOAT): API 호출 소요 시간
- **토큰 사용량 정보** (모든 LLM 호출의 토큰 정보를 통합 관리):
  - `prompt_tokens` (INTEGER): 프롬프트 토큰 수 (입력)
  - `completion_tokens` (INTEGER): 생성 토큰 수 (출력)
  - `total_tokens` (INTEGER): 총 토큰 수
  - `token_usage` (JSONB): 토큰 사용량 정보 원본 (예: `{"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300}`)
- `created_at`, `updated_at`

---

## 🔧 구현해야 할 API 엔드포인트

### 1. `/api/js/gpt/kor-to-eng` (신규 생성)

**목적**: 한국어 설명을 영어로 변환

**요청 (Request):**
```json
{
  "job_id": "uuid-string",
  "tenant_id": "string"
}
```

**처리 과정:**
1. `job_inputs` 테이블에서 `desc_kor` 조회
2. GPT API 호출: 한국어 → 영어 변환
3. GPT API 호출 후 토큰 사용량 추출 및 `llm_traces` 테이블에 기록:
   ```sql
   INSERT INTO llm_traces (
       llm_trace_id, job_id, provider, llm_model_id, operation_type,
       request, response, latency_ms,
       prompt_tokens, completion_tokens, total_tokens, token_usage,
       created_at, updated_at
   ) VALUES (
       :llm_trace_id, :job_id, 'gpt', :llm_model_id, 'kor_to_eng',
       CAST(:request AS jsonb), CAST(:response AS jsonb), :latency_ms,
       :prompt_tokens, :completion_tokens, :total_tokens, CAST(:token_usage AS jsonb),
       CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
   )
   ```
   - **LLM 모델 ID**: `llm_models` 테이블에서 사용한 모델 조회 (선택적, NULL 허용)
   - **토큰 정보 추출**: GPT API 응답에서 `token_usage` 추출
     - `prompt_tokens`: `token_usage.prompt_tokens`
     - `completion_tokens`: `token_usage.completion_tokens`
     - `total_tokens`: `token_usage.total_tokens`
     - `token_usage`: 원본 JSON 객체
4. `txt_ad_copy_generations` 테이블에 레코드 생성:
   ```sql
   INSERT INTO txt_ad_copy_generations (
       ad_copy_gen_id, job_id, llm_trace_id, generation_stage,
       ad_copy_eng, status, created_at, updated_at
   ) VALUES (
       :ad_copy_gen_id, :job_id, :llm_trace_id, 'kor_to_eng',
       :ad_copy_eng, 'done', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
   )
   ```
5. `job_inputs.desc_eng` 업데이트 (영어 설명으로)
6. `jobs` 테이블 업데이트:
   ```sql
   UPDATE jobs
   SET current_step = 'desc_kor_translate',
       status = 'done',
       updated_at = CURRENT_TIMESTAMP
   WHERE job_id = :job_id
   ```

**응답 (Response):**
```json
{
  "job_id": "uuid-string",
  "llm_trace_id": "uuid-string",
  "ad_copy_gen_id": "uuid-string",
  "desc_eng": "English description",
  "status": "done"
}
```

---

### 2. `/api/js/gpt/ad-copy-eng` (신규 생성)

**목적**: 영어 광고문구 생성

**요청 (Request):**
```json
{
  "job_id": "uuid-string",
  "tenant_id": "string"
}
```

**처리 과정:**
1. `job_inputs` 테이블에서 다음 데이터 조회:
   - `desc_eng`: 영어 설명 (kor_to_eng 결과)
   - `tone_style_id`: 톤 & 스타일 ID
2. `tone_styles` 테이블에서 톤 & 스타일 정보 조회
3. GPT API 호출: 영어 광고문구 생성
   - 입력: `desc_eng`, `tone_style` 정보
   - 출력: 영어 광고문구
4. GPT API 호출 후 토큰 사용량 추출 및 `llm_traces` 테이블에 기록:
   ```sql
   INSERT INTO llm_traces (
       llm_trace_id, job_id, provider, llm_model_id, operation_type,
       request, response, latency_ms,
       prompt_tokens, completion_tokens, total_tokens, token_usage,
       created_at, updated_at
   ) VALUES (
       :llm_trace_id, :job_id, 'gpt', :llm_model_id, 'ad_copy_gen',
       CAST(:request AS jsonb), CAST(:response AS jsonb), :latency_ms,
       :prompt_tokens, :completion_tokens, :total_tokens, CAST(:token_usage AS jsonb),
       CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
   )
   ```
   - **LLM 모델 ID**: `llm_models` 테이블에서 사용한 모델 조회 (선택적, NULL 허용)
   - **토큰 정보 추출**: GPT API 응답에서 `token_usage` 추출
     - `prompt_tokens`: `token_usage.prompt_tokens`
     - `completion_tokens`: `token_usage.completion_tokens`
     - `total_tokens`: `token_usage.total_tokens`
     - `token_usage`: 원본 JSON 객체
5. `txt_ad_copy_generations` 테이블에 레코드 생성/업데이트:
   ```sql
   INSERT INTO txt_ad_copy_generations (
       ad_copy_gen_id, job_id, llm_trace_id, generation_stage,
       ad_copy_eng, status, created_at, updated_at
   ) VALUES (
       :ad_copy_gen_id, :job_id, :llm_trace_id, 'ad_copy_eng',
       :ad_copy_eng, 'done', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
   )
   ON CONFLICT (job_id, generation_stage) 
   DO UPDATE SET 
       ad_copy_eng = EXCLUDED.ad_copy_eng,
       llm_trace_id = EXCLUDED.llm_trace_id,
       status = 'done',
       updated_at = CURRENT_TIMESTAMP
   ```
6. `jobs` 테이블 업데이트:
   ```sql
   UPDATE jobs -> llm trace
   SET current_step = 'ad_copy_gen_eng',
       status = 'done',
       updated_at = CURRENT_TIMESTAMP
   WHERE job_id = :job_id
   ```

**응답 (Response):**
```json
{
  "job_id": "uuid-string",
  "llm_trace_id": "uuid-string",
  "ad_copy_gen_id": "uuid-string",
  "ad_copy_eng": "English ad copy text",
  "status": "done"
}
```

---

### 3. `/api/js/gpt/ad-copy-kor` (신규 생성)

**목적**: 한글 광고문구 생성 (오버레이에 사용)

**요청 (Request):**
```json
{
  "job_id": "uuid-string",
  "tenant_id": "string"
}
```

**처리 과정:**
1. `txt_ad_copy_generations` 테이블에서 `ad_copy_eng` 조회 (generation_stage='ad_copy_eng')
2. GPT API 호출: 영어 광고문구 → 한글 광고문구 변환
3. GPT API 호출 후 토큰 사용량 추출 및 `llm_traces` 테이블에 기록:
   ```sql
   INSERT INTO llm_traces (
       llm_trace_id, job_id, provider, llm_model_id, operation_type,
       request, response, latency_ms,
       prompt_tokens, completion_tokens, total_tokens, token_usage,
       created_at, updated_at
   ) VALUES (
       :llm_trace_id, :job_id, 'gpt', :llm_model_id, 'ad_copy_kor',
       CAST(:request AS jsonb), CAST(:response AS jsonb), :latency_ms,
       :prompt_tokens, :completion_tokens, :total_tokens, CAST(:token_usage AS jsonb),
       CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
   )
   ```
   - **LLM 모델 ID**: `llm_models` 테이블에서 사용한 모델 조회 (선택적, NULL 허용)
   - **토큰 정보 추출**: GPT API 응답에서 `token_usage` 추출
     - `prompt_tokens`: `token_usage.prompt_tokens`
     - `completion_tokens`: `token_usage.completion_tokens`
     - `total_tokens`: `token_usage.total_tokens`
     - `token_usage`: 원본 JSON 객체
4. `txt_ad_copy_generations` 테이블에 레코드 생성:
   ```sql
   INSERT INTO txt_ad_copy_generations (
       ad_copy_gen_id, job_id, llm_trace_id, generation_stage,
       ad_copy_kor, status, created_at, updated_at
   ) VALUES (
       :ad_copy_gen_id, :job_id, :llm_trace_id, 'ad_copy_kor',
       :ad_copy_kor, 'done', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
   )
   ```
5. `jobs` 테이블 업데이트:
   ```sql
   UPDATE jobs -> llm trance
   SET current_step = 'ad_copy_gen_kor',
       status = 'done',
       updated_at = CURRENT_TIMESTAMP
   WHERE job_id = :job_id
   ```

**응답 (Response):**
```json
{
  "job_id": "uuid-string",
  "llm_trace_id": "uuid-string",
  "ad_copy_gen_id": "uuid-string",
  "ad_copy_kor": "한글 광고문구 텍스트",
  "status": "done"
}
```

**⚠️ 중요**: 
- 이 한글 광고문구(`ad_copy_kor`)는 YH 파트의 `overlay` 단계에서 오버레이 텍스트로 사용됩니다.
- YH 파트는 `ad_copy_kor`를 우선적으로 사용하며, 없을 경우 `ad_copy_eng`를 fallback으로 사용합니다.

---

## 📝 구현 체크리스트

### 1. 데이터베이스 연결
- [ ] `job_inputs` 테이블 조회 구현
- [ ] `txt_ad_copy_generations` 테이블 INSERT 구현
- [ ] `llm_traces` 테이블 INSERT 구현
- [ ] `jobs` 테이블 UPDATE 구현

### 2. GPT API 연동
- [ ] GPT API 클라이언트 설정
- [ ] 한국어 → 영어 변환 프롬프트 작성
- [ ] 영어 광고문구 생성 프롬프트 작성
- [ ] 영어 → 한글 광고문구 변환 프롬프트 작성
- [ ] 에러 처리 및 재시도 로직

### 3. Trace 관리
- [ ] `llm_models` 테이블에서 사용할 모델 조회 (선택적)
- [ ] `llm_traces` 테이블에 요청/응답 저장
- [ ] `llm_model_id` 저장 (사용한 모델이 있는 경우)
- [ ] `latency_ms` 측정 및 저장
- [ ] `operation_type` 올바르게 설정
- [ ] **토큰 사용량 정보 저장** (GPT API 응답에서 추출):
  - `prompt_tokens`, `completion_tokens`, `total_tokens` 저장
  - `token_usage` JSONB 원본 저장

### 4. 데이터 흐름
- [ ] `kor_to_eng` 완료 후 `ad_copy_eng` 자동 실행 여부 확인
- [ ] `ad_copy_eng` 완료 후 `ad_copy_kor` 자동 실행 여부 확인
- [ ] `txt_ad_copy_generations` 레코드 생성 확인 (kor_to_eng, ad_copy_eng, ad_copy_kor)
- [ ] `job_inputs.desc_eng` 업데이트 확인

---

## 🔗 YH 파트와의 연동

### 데이터 공유
- **JS 파트가 생성한 데이터**: `txt_ad_copy_generations` 테이블에 저장
  - `generation_stage='kor_to_eng'`: 영어 설명
  - `generation_stage='ad_copy_eng'`: 영어 광고문구
  - `generation_stage='ad_copy_kor'`: 한글 광고문구 (오버레이에 사용)
- **YH 파트가 사용하는 데이터**: `txt_ad_copy_generations` 테이블 조회
  - `vlm_analyze` 단계: `ad_copy_eng` 사용
  - `overlay` 단계: `ad_copy_kor` 우선 사용, 없으면 `ad_copy_eng` fallback
  - `eng_to_kor` 단계: `refined_ad_copy_eng` 또는 `ad_copy_eng` 사용

### 실행 시점
- **`kor_to_eng`**: Job 생성 직후 또는 `img_gen` 전 실행
- **`ad_copy_eng`**: `kor_to_eng` 완료 후 실행
- **`ad_copy_kor`**: `ad_copy_eng` 완료 후 실행 (YH 파트 시작 전에 완료되어야 함)

---

## ❓ 질문 및 문의

구현 중 문제가 발생하거나 질문이 있으면 YH 파트 담당자에게 문의하세요.

