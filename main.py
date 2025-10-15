import os
import base64
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
from ollama import Client
from pydantic import BaseModel
import uvicorn
import uuid
import requests
import logging

from web_ui import get_web_ui_html
from forge_proxy import forge_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 環境変数設定
DEFAULT_TRANSLATE_MODEL = os.getenv("TRANSLATE_MODEL", "brxce/stable-diffusion-prompt-generator:latest")
DEFAULT_FORGE_MODEL = os.getenv("FORGE_MODEL", "sd\\novaAnimeXL_ilV5b.safetensors")

PROMPT_TEMPLATE = os.getenv(
    "TRANSLATE_PROMPT",
    "Translate the following Japanese text into concise, comma-separated English tags optimized for Stable Diffusion prompts in Danbooru style. Return only the translated tags, without any additional descriptions, styles, or metadata: {text}"
)

TRANSLATE_HOST = os.getenv("TRANSLATE_HOST", "192.168.2.199")
TRANSLATE_PORT = int(os.getenv("TRANSLATE_PORT", "8091"))

# Ollama設定
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "192.168.2.197")
OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")
OLLAMA_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

# Forge API設定
FORGE_HOST = os.getenv("FORGE_HOST", "192.168.2.197")
FORGE_PORT = os.getenv("FORGE_PORT", "7865")
FORGE_URL = f"http://{FORGE_HOST}:{FORGE_PORT}"

# 画像保存設定
SAVE_DIR = os.getenv("SAVE_DIR", "/app/images")
CONFIG_DIR = "/app/config"
CONFIG_FILE = os.path.join(CONFIG_DIR, "config_Jp2Prompt.json")

# ディレクトリ作成
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

logger.info(f"Using OLLAMA: {OLLAMA_URL}")
logger.info(f"Using FORGE: {FORGE_URL}")
logger.info(f"Image save directory: {SAVE_DIR}")
logger.info(f"Config file: {CONFIG_FILE}")

# ================================
# Pydantic モデル定義
# ================================

class TranslateRequest(BaseModel):
    japanese_prompt: str
    model: str = DEFAULT_TRANSLATE_MODEL
    context_id: Optional[str] = None
    session_id: Optional[str] = None

class GetImageRequest(BaseModel):
    japanese_prompt: str
    negative_prompt: str = ""
    translate_model: str = DEFAULT_TRANSLATE_MODEL  # 翻訳用モデル
    # 基本パラメータ
    width: int = 512
    height: int = 512
    cfg_scale: float = 7.0
    steps: int = 20
    batch_size: int = 1
    batch_count: int = 1
    # オプション設定
    selected_model: Optional[str] = None
    selected_vae: Optional[str] = None
    selected_text_encoder: Optional[str] = None
    selected_unet: Optional[str] = None
    dynamic_prompts: bool = False

class ConfigUpdateRequest(BaseModel):
    # モデル設定
    sd_model_checkpoint: Optional[str] = None
    sd_vae: Optional[str] = None
    # VAE/Text Encoder設定 (Flux用)
    selected_modules: Optional[Dict[str, str]] = None
    # 画像生成設定
    default_width: Optional[int] = None
    default_height: Optional[int] = None
    default_cfg_scale: Optional[float] = None
    default_steps: Optional[int] = None
    default_batch_size: Optional[int] = None
    default_batch_count: Optional[int] = None
    # Dynamic Prompts設定
    dynamic_prompts_enabled: Optional[bool] = None
    # プロンプト設定
    default_prompt: Optional[str] = None
    default_negative_prompt: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    message: str

class ConfigResponse(BaseModel):
    current_config: Dict[str, Any]
    available_models: List[str]
    available_vaes: List[str] 
    available_modules: List[Dict[str, str]]
    config_history: List[Dict[str, Any]]

# ================================
# グローバル設定管理
# ================================

class ConfigManager:
    def __init__(self):
        self.default_config = {
            "sd_model_checkpoint": DEFAULT_FORGE_MODEL,
            "sd_vae": "Automatic",
            "selected_modules": {},
            "default_width": 512,
            "default_height": 512,
            "default_cfg_scale": 7.0,
            "default_steps": 20,
            "default_batch_size": 1,
            "default_batch_count": 1,
            "dynamic_prompts_enabled": False,
            "default_prompt": "masterpiece, best quality, highly detailed, ",
            "default_negative_prompt": "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, "
        }
        self.current_config = self.default_config.copy()
        self.config_history = []
        self.load_config()
        
    def load_config(self):
        """設定ファイルから設定を読み込み"""
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    saved_config = json.load(f)
                    loaded_config = saved_config.get('current_config', {})
                    
                    # デフォルト値で初期化した後、保存された設定で上書き
                    for key, value in loaded_config.items():
                        if key in self.default_config:
                            self.current_config[key] = value
                    
                    # 新しい設定項目が追加された場合のマイグレーション
                    config_updated = False
                    for key, default_value in self.default_config.items():
                        if key not in self.current_config:
                            self.current_config[key] = default_value
                            config_updated = True
                            logger.info(f"Added new config key: {key} = {default_value}")
                    
                    if config_updated:
                        self.save_config()
                    
                    self.config_history = saved_config.get('config_history', [])
                logger.info(f"Config loaded from {CONFIG_FILE}")
            else:
                logger.info("No config file found, using defaults")
                self.save_config()
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            self.current_config = self.default_config.copy()
            
    def save_config(self):
        """設定ファイルに保存"""
        try:
            config_data = {
                "current_config": self.current_config,
                "config_history": self.config_history,
                "last_updated": datetime.now().isoformat()
            }
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Config saved to {CONFIG_FILE}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
        
    def update_config(self, new_config: Dict[str, Any]):
        """設定を更新"""
        # 履歴に現在の設定を保存
        self.config_history.append({
            "timestamp": datetime.now().isoformat(),
            "config": self.current_config.copy()
        })
        
        # 履歴は最新10件まで保持
        if len(self.config_history) > 10:
            self.config_history = self.config_history[-10:]
            
        # 設定を更新
        for key, value in new_config.items():
            if value is not None and key in self.current_config:
                self.current_config[key] = value
                
        # ファイルに保存
        self.save_config()
        logger.info(f"Config updated: {new_config}")
        
    def get_config(self):
        return self.current_config.copy()
        
    def get_history(self):
        return self.config_history.copy()

config_manager = ConfigManager()

# ================================
# FastAPIアプリケーション設定
# ================================

app = FastAPI(
    title="Japanese to English Translation & Image Generation API",
    description="API for translating Japanese prompts and generating images using Ollama LLMs and Automatic1111 Forge.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

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
# Helper Functions
# ================================

async def get_forge_models():
    """Forgeから利用可能なモデル一覧を取得"""
    try:
        response = requests.get(f"{FORGE_URL}/sdapi/v1/sd-models", timeout=10)
        if response.status_code == 200:
            models = response.json()
            return [model["title"] for model in models]
        else:
            logger.warning(f"Failed to fetch models from Forge: {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        return []

async def get_forge_vaes():
    """ForgeからVAE一覧を取得"""
    try:
        response = requests.get(f"{FORGE_URL}/sdapi/v1/options", timeout=10)
        if response.status_code == 200:
            return ["Automatic", "None"]
        else:
            return ["Automatic", "None"]
    except Exception as e:
        logger.error(f"Error fetching VAEs: {e}")
        return ["Automatic", "None"]

async def get_forge_modules():
    """ForgeからSD Modules（Text Encoder, UNET等）一覧を取得"""
    try:
        response = requests.get(f"{FORGE_URL}/sdapi/v1/sd-modules", timeout=10)
        if response.status_code == 200:
            modules = response.json()
            return [{"model_name": mod["model_name"], "filename": mod["filename"]} for mod in modules]
        else:
            logger.warning(f"Failed to fetch modules from Forge: {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"Error fetching modules: {e}")
        return []

async def translate_japanese_prompt(japanese_text: str, model: str = DEFAULT_TRANSLATE_MODEL):
    """日本語プロンプトを英語に翻訳"""
    try:
        logger.info(f"Translating text with model {model}")
        ollama_client = Client(host=OLLAMA_URL)
        prompt_text = PROMPT_TEMPLATE.format(text=japanese_text)
        
        response = ollama_client.generate(
            model=model,
            prompt=prompt_text,
            options={"temperature": 0.5}
        )
        
        translated = response["response"].strip()
        logger.info(f"Translation completed: {japanese_text[:50]}... -> {translated[:50]}...")
        return translated
        
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")

async def generate_image_with_forge(translated_prompt: str, params: Dict[str, Any], config: Dict[str, Any]):
    """Forge APIを使用して画像生成"""
    try:
        # デフォルトプロンプトと翻訳されたプロンプトを結合
        default_prompt = config.get("default_prompt", "")
        default_negative_prompt = config.get("default_negative_prompt", "")
        
        # プロンプトを結合し重複除去
        final_prompt = combine_prompts(default_prompt, translated_prompt)
        final_negative_prompt = combine_prompts(default_negative_prompt, params.get("negative_prompt", ""))
        
        forge_params = {
            "prompt": final_prompt,
            "negative_prompt": final_negative_prompt,
            "width": params.get("width", 512),
            "height": params.get("height", 512),
            "cfg_scale": params.get("cfg_scale", 7.0),
            "steps": params.get("steps", 20),
            "batch_size": params.get("batch_size", 1),
            "n_iter": params.get("batch_count", 1),
            "sampler_name": "Euler a",
            "save_images": False,
            "send_images": True,
        }
        
        # Dynamic Prompts設定
        if params.get("dynamic_prompts", False):
            forge_params["alwayson_scripts"] = {
                "Dynamic Prompts": {
                    "args": [True]
                }
            }
        
        # Forge APIのオーバーライド設定でモデル等を指定
        override_settings = {}
        
        if params.get("selected_model"):
            override_settings["sd_model_checkpoint"] = params["selected_model"]
        elif config.get("sd_model_checkpoint"):
            override_settings["sd_model_checkpoint"] = config["sd_model_checkpoint"]
            
        if params.get("selected_vae"):
            override_settings["sd_vae"] = params["selected_vae"]
        elif config.get("sd_vae"):
            override_settings["sd_vae"] = config["sd_vae"]
            
        if params.get("selected_text_encoder"):
            override_settings["sd_text_encoder"] = params["selected_text_encoder"]
            
        if params.get("selected_unet"):
            override_settings["sd_unet"] = params["selected_unet"]
        
        if override_settings:
            forge_params["override_settings"] = override_settings
            forge_params["override_settings_restore_afterwards"] = True
        
        logger.info(f"Final prompt: {final_prompt[:100]}...")
        logger.info(f"Final negative prompt: {final_negative_prompt[:100]}...")
        logger.info(f"Generating image with Forge API: {forge_params}")
        
        response = requests.post(
            f"{FORGE_URL}/sdapi/v1/txt2img",
            json=forge_params,
            timeout=600
        )
        
        if response.status_code == 200:
            result = response.json()
            # 結果にプロンプト情報を追加
            result["final_prompt"] = final_prompt
            result["final_negative_prompt"] = final_negative_prompt
            return result
        else:
            logger.error(f"Forge API error: {response.status_code} - {response.text}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Forge API error: {response.text}"
            )
            
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Image generation timeout")
    except Exception as e:
        logger.error(f"Image generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image generation error: {str(e)}")

async def save_image_to_nas(image_base64: str, filename: str):
    """Base64画像をNASに保存"""
    try:
        image_data = base64.b64decode(image_base64)
        filepath = os.path.join(SAVE_DIR, filename)
        
        # ディレクトリの権限確認・修正
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR, mode=0o755, exist_ok=True)
            logger.info(f"Created directory: {SAVE_DIR}")
        
        # ファイル保存前に権限確認
        try:
            with open(filepath, "wb") as f:
                f.write(image_data)
            logger.info(f"Image saved successfully: {filepath}")
        except PermissionError as pe:
            # 権限エラーの場合、詳細ログを出力
            import pwd
            import grp
            import stat
            
            dir_stat = os.stat(SAVE_DIR)
            current_user = pwd.getpwuid(os.getuid()).pw_name
            current_group = grp.getgrgid(os.getgid()).gr_name
            dir_owner = pwd.getpwuid(dir_stat.st_uid).pw_name
            dir_group = grp.getgrgid(dir_stat.st_gid).gr_name
            dir_permissions = oct(stat.S_IMODE(dir_stat.st_mode))
            
            logger.error(f"Permission denied details:")
            logger.error(f"  Current user: {current_user}({os.getuid()})")
            logger.error(f"  Current group: {current_group}({os.getgid()})")
            logger.error(f"  Directory: {SAVE_DIR}")
            logger.error(f"  Directory owner: {dir_owner}({dir_stat.st_uid})")
            logger.error(f"  Directory group: {dir_group}({dir_stat.st_gid})")
            logger.error(f"  Directory permissions: {dir_permissions}")
            
            raise HTTPException(
                status_code=500, 
                detail=f"Permission denied: Cannot write to {SAVE_DIR}. Current user: {current_user}, Directory owner: {dir_owner}, Permissions: {dir_permissions}"
            )
            
        return filepath
        
    except Exception as e:
        logger.error(f"Error saving image: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving image: {str(e)}")

def remove_duplicate_tags(prompt: str) -> str:
    """プロンプトから重複タグを除去"""
    # カンマ区切りでタグを分割
    tags = [tag.strip() for tag in prompt.split(',') if tag.strip()]
    
    # 重複除去（順序を保持）
    seen = set()
    unique_tags = []
    for tag in tags:
        tag_lower = tag.lower()
        if tag_lower not in seen:
            seen.add(tag_lower)
            unique_tags.append(tag)
    
    return ', '.join(unique_tags)

def combine_prompts(default_prompt: str, user_prompt: str) -> str:
    """デフォルトプロンプトとユーザープロンプトを結合し、重複を除去"""
    combined = f"{default_prompt.strip()}, {user_prompt.strip()}"
    return remove_duplicate_tags(combined)

# ================================
# 基本エンドポイント
# ================================

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("favicon.ico")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        message="Translation & Image Generation API is running"
    )

# ================================
# 設定UI エンドポイント
# ================================

@app.get("/", response_class=HTMLResponse)
async def get_settings_ui():
    """設定用HTML UIを返す"""
    return HTMLResponse(content=get_web_ui_html())

# ================================
# 静的ファイル配信用のエンドポイント
# ================================

@app.get("/static/animated.gif", include_in_schema=False)
async def get_animated_gif():
    """ローディング用アニメーションGIFを配信"""
    gif_path = os.path.join(os.path.dirname(__file__), "animated.gif")
    if os.path.exists(gif_path):
        return FileResponse(gif_path, media_type="image/gif")
    else:
        # ファイルが存在しない場合はプレースホルダーGIFを返す
        raise HTTPException(status_code=404, detail="animated.gif not found")

@app.get("/static/err.gif", include_in_schema=False)
async def get_error_gif():
    """エラー用GIFを配信"""
    gif_path = os.path.join(os.path.dirname(__file__), "err.gif")
    if os.path.exists(gif_path):
        return FileResponse(gif_path, media_type="image/gif")
    else:
        # ファイルが存在しない場合はプレースホルダーGIFを返す
        raise HTTPException(status_code=404, detail="err.gif not found")

# ================================
# メインAPI エンドポイント
# ================================

@app.post("/translate", tags=["translation"])
async def translate_jp_to_en(request: TranslateRequest):
    """日本語プロンプトを英語に翻訳（既存機能）"""
    context_id = request.context_id or str(uuid.uuid4())
    session_id = request.session_id

    if session_id:
        _context_store.setdefault(session_id, {})[context_id] = {"prompt": request.japanese_prompt}

    translated = await translate_japanese_prompt(request.japanese_prompt, request.model)
    
    if session_id:
        _context_store[session_id][context_id]["translated"] = translated

    return {
        "translated_prompt": translated,
        "context_id": context_id,
        "session_id": session_id
    }

@app.post("/get_image", tags=["image_generation"])
async def get_image(request: GetImageRequest):
    """日本語プロンプトから画像生成"""
    try:
        logger.info(f"Starting image generation for: {request.japanese_prompt[:50]}...")
        translated_prompt = await translate_japanese_prompt(
            request.japanese_prompt, 
            request.translate_model
        )
        
        current_config = config_manager.get_config()
        
        params = {
            "negative_prompt": request.negative_prompt,
            "width": request.width or current_config.get("default_width", 512),
            "height": request.height or current_config.get("default_height", 512),
            "cfg_scale": request.cfg_scale or current_config.get("default_cfg_scale", 7.0),
            "steps": request.steps or current_config.get("default_steps", 20),
            "batch_size": request.batch_size or current_config.get("default_batch_size", 1),
            "batch_count": request.batch_count or current_config.get("default_batch_count", 1),
            "dynamic_prompts": request.dynamic_prompts or current_config.get("dynamic_prompts_enabled", False),
            "selected_model": request.selected_model,
            "selected_vae": request.selected_vae,
            "selected_text_encoder": request.selected_text_encoder,
            "selected_unet": request.selected_unet,
        }
        
        forge_response = await generate_image_with_forge(translated_prompt, params, current_config)
        
        saved_files = []
        if "images" in forge_response and forge_response["images"]:
            for i, image_base64 in enumerate(forge_response["images"]):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"generated_{timestamp}_{i:03d}.png"
                filepath = await save_image_to_nas(image_base64, filename)
                saved_files.append(filepath)
        
        return {
            "translated_prompt": translated_prompt,
            "images": forge_response.get("images", []),
            "saved_files": saved_files,
            "parameters": params,
            "info": forge_response.get("info", ""),
            "generation_info": {
                "original_prompt": request.japanese_prompt,
                "translated_prompt": translated_prompt,
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Image generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

@app.get("/config", response_model=ConfigResponse)
async def get_config():
    """現在の設定と利用可能なオプションを取得"""
    try:
        available_models = await get_forge_models()
        available_vaes = await get_forge_vaes()
        available_modules = await get_forge_modules()
        
        return ConfigResponse(
            current_config=config_manager.get_config(),
            available_models=available_models,
            available_vaes=available_vaes,
            available_modules=available_modules,
            config_history=config_manager.get_history()
        )
    except Exception as e:
        logger.error(f"Config retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Config retrieval failed: {str(e)}")

@app.post("/config", tags=["configuration"])
async def update_config(request: ConfigUpdateRequest):
    """設定を更新"""
    try:
        # Forge APIの設定を更新（モデル切り替え等）
        forge_updates = {}
        
        if request.sd_model_checkpoint:
            forge_updates["sd_model_checkpoint"] = request.sd_model_checkpoint
            
        if request.sd_vae:
            forge_updates["sd_vae"] = request.sd_vae
            
        # Forge APIに設定を送信
        if forge_updates:
            try:
                response = requests.post(
                    f"{FORGE_URL}/sdapi/v1/options",
                    json=forge_updates,
                    timeout=30
                )
                if response.status_code != 200:
                    logger.warning(f"Forge config update failed: {response.status_code} - {response.text}")
            except Exception as e:
                logger.warning(f"Failed to update Forge config: {e}")
        
        # ローカル設定を更新
        update_data = {}
        for field, value in request.dict().items():
            if value is not None:
                update_data[field] = value
                
        config_manager.update_config(update_data)
        
        return {
            "message": "Configuration updated successfully",
            "updated_config": config_manager.get_config()
        }
        
    except Exception as e:
        logger.error(f"Config update failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Config update failed: {str(e)}")

@app.post("/config/reset", tags=["configuration"])
async def reset_config():
    """設定をデフォルト値にリセット"""
    try:
        config_manager.current_config = config_manager.default_config.copy()
        config_manager.save_config()
        
        return {
            "message": "Configuration reset to default values",
            "config": config_manager.get_config()
        }
        
    except Exception as e:
        logger.error(f"Config reset failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Config reset failed: {str(e)}")

# ================================
# Forge routerをメインアプリに追加
# ================================
app.include_router(forge_router)

# ================================
# システム情報エンドポイント
# ================================

@app.get("/system/info", tags=["system"])
async def get_system_info():
    """システム情報とForge接続状態を取得"""
    try:
        # Forge API接続テスト
        forge_status = "disconnected"
        forge_info = {}
        
        try:
            response = requests.get(f"{FORGE_URL}/sdapi/v1/options", timeout=10)
            if response.status_code == 200:
                forge_status = "connected"
                forge_info = {"version": "Available", "status": "OK"}
        except Exception:
            pass
            
        # Ollama接続テスト
        ollama_status = "disconnected"
        try:
            ollama_client = Client(host=OLLAMA_URL)
            response = ollama_client.generate(
                model=DEFAULT_TRANSLATE_MODEL,
                prompt="test",
                options={"temperature": 0.1}
            )
            if response:
                ollama_status = "connected"
        except Exception:
            pass
            
        return {
            "status": "running",
            "forge_connection": {
                "status": forge_status,
                "url": FORGE_URL,
                "info": forge_info
            },
            "ollama_connection": {
                "status": ollama_status,
                "url": OLLAMA_URL,
                "model": DEFAULT_TRANSLATE_MODEL
            },
            "storage": {
                "save_directory": SAVE_DIR,
                "config_directory": CONFIG_DIR,
                "available": os.path.exists(SAVE_DIR)
            },
            "config": config_manager.get_config()
        }
        
    except Exception as e:
        logger.error(f"System info retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"System info retrieval failed: {str(e)}")

@app.get("/images", tags=["files"])
async def list_generated_images():
    """生成された画像ファイルの一覧を取得"""
    try:
        if not os.path.exists(SAVE_DIR):
            return {"images": [], "message": "Image directory not found"}
            
        image_files = []
        for filename in os.listdir(SAVE_DIR):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(SAVE_DIR, filename)
                stat = os.stat(filepath)
                image_files.append({
                    "filename": filename,
                    "path": filepath,
                    "size": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
                
        # 作成日時でソート（新しい順）
        image_files.sort(key=lambda x: x["created"], reverse=True)
        
        return {
            "images": image_files,
            "total_count": len(image_files)
        }
        
    except Exception as e:
        logger.error(f"Image list retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image list retrieval failed: {str(e)}")

@app.get("/images/{filename}", tags=["files"])
async def get_image_file(filename: str):
    """指定された画像ファイルを返す"""
    try:
        filepath = os.path.join(SAVE_DIR, filename)
        
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="Image file not found")
            
        return FileResponse(filepath)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image file retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image file retrieval failed: {str(e)}")

# ================================
# OpenAPI設定
# ================================

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
        
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=[route for route in app.routes],
        servers=[{
            "url": f"http://{TRANSLATE_HOST}:{TRANSLATE_PORT}", 
            "description": "Translation & Image Generation API Server"
        }],
    )
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# ================================
# 起動設定
# ================================

if __name__ == "__main__":
    logger.info("Starting Translation & Image Generation API...")
    logger.info(f"Server: http://{TRANSLATE_HOST}:{TRANSLATE_PORT}")
    logger.info(f"Web UI: http://{TRANSLATE_HOST}:{TRANSLATE_PORT}/")
    logger.info(f"Ollama: {OLLAMA_URL}")
    logger.info(f"Forge: {FORGE_URL}")
    logger.info(f"Save directory: {SAVE_DIR}")
    logger.info(f"Config file: {CONFIG_FILE}")
    logger.info(f"Forge Proxy: http://{TRANSLATE_HOST}:{TRANSLATE_PORT}/sd/")
    
    uvicorn.run(
        app, 
        host=TRANSLATE_HOST, 
        port=TRANSLATE_PORT,
        log_level="info"
    )