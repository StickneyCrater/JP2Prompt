import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
from ollama import Client
from pydantic import BaseModel
import uvicorn
import uuid
from typing import Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("TRANSLATE_MODEL", "brxce/stable-diffusion-prompt-generator:latest")
PROMPT_TEMPLATE = os.getenv(
    "TRANSLATE_PROMPT",
    "Translate the following Japanese text to danbooru English optimized for Stable Diffusion prompts: {text}"
)
TRANSLATE_HOST = os.getenv("TRANSLATE_HOST", "192.168.2.199")
TRANSLATE_PORT = os.getenv("TRANSLATE_PORT", "8091")

# Ollamaホスト
OLLAMA_HOST = os.getenv("OLLAMA_URL", "192.168.2.197")
OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://" + OLLAMA_HOST + ":" + OLLAMA_PORT)
logger.info(f"Using OLLAMA: {OLLAMA_URL}")


# ================================
# FastAPIアプリケーション設定
# ================================
# プロジェクトに合わせてタイトルや説明をカスタマイズしてください
app = FastAPI(
    title="Japanese to English Translation API",
    description="API for translating Japanese prompts to English using Ollama LLMs.",
    version="1.0.0",  # 👈 バージョンを管理
    docs_url="/docs",  # Swagger UIのパス（無効化する場合はNone）
    redoc_url="/redoc",  # ReDocのパス（無効化する場合はNone）
)


# ================================
# Pydantic モデル定義
# ================================
# データバリデーション用のモデルをここに追加
class TranslateRequest(BaseModel):
    japanese_prompt: str
    model: str = DEFAULT_MODEL
    context_id: Optional[str] = None
    session_id: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    message: str

# ================================
# 基本エンドポイント
# ================================

# favicon.ico エンドポイント
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("favicon.ico")

# ヘルスチェックエンドポイント（Docker用）
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        message="ARM v7 FastAPI Base is running"  # 👈 メッセージをカスタマイズ
    )

# ルートエンドポイント
@app.get("/")
async def root():
    return {
        "message": "ARM v7 Alpine FastAPI Base Image",  # 👈 ウェルカムメッセージを変更
        "project": "armv7-fastapi-base",  # 👈 プロジェクト名を設定
        "version": "1.0.0"
    }

# ================================
# カスタムエンドポイント
# ================================
# ここにプロジェクト固有のエンドポイントを追加してください

@app.post("/translate", tags=["translation"])
async def translate_jp_to_en(request: TranslateRequest):
    context_id = request.context_id or str(uuid.uuid4())
    session_id = request.session_id

    if session_id:
        _context_store.setdefault(session_id, {})[context_id] = {"prompt": request.japanese_prompt}

    try:
        logger.info(f"Connecting to Ollama at {OLLAMA_URL}")
        ollama_client = Client(host=OLLAMA_URL)
        prompt_text = PROMPT_TEMPLATE.format(text=request.japanese_prompt)
        response = ollama_client.generate(
            model=request.model,
            prompt=prompt_text,
            options={"temperature": 0.5}
        )
        translated = response["response"].strip()

        if session_id:
            _context_store[session_id][context_id]["translated"] = translated

        return {
            "translated_prompt": translated,
            "context_id": context_id,
            "session_id": session_id
        }
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")

# ================================
# ミドルウェア設定（オプション）
# ================================
# CORS、認証、ログ設定などをここに追加

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_context_store: Dict[str, Dict[str, Any]] = {}

# ================================
# 起動設定
# ================================

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=[route for route in app.routes],
        servers=[{"url": f"http://{TRANSLATE_HOST}:{TRANSLATE_PORT}", "description": app.title + " API server."}],
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi


if __name__ == "__main__":
    # 開発用の起動設定
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=TRANSLATE_PORT,
        # reload=True,  # 開発時のみ有効にする
        log_level="debug"  # 本番では"info"に変更
    )