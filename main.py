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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 環境変数設定
DEFAULT_TRANSLATE_MODEL = os.getenv("TRANSLATE_MODEL", "brxce/stable-diffusion-prompt-generator:latest")
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
DEFAULT_FORGE_MODEL = os.getenv("FORGE_MODEL", "sd\\ovaAnimeXL_ilV5b.safetensors")
FORGE_HOST = os.getenv("FORGE_HOST", "127.0.0.1")
FORGE_PORT = os.getenv("FORGE_PORT", "7860")
FORGE_URL = f"http://{FORGE_HOST}:{FORGE_PORT}"

# 画像保存設定
SAVE_DIR = os.getenv("SAVE_DIR", "/app/images")
os.makedirs(SAVE_DIR, exist_ok=True)

logger.info(f"Using OLLAMA: {OLLAMA_URL}")
logger.info(f"Using FORGE: {FORGE_URL}")
logger.info(f"Image save directory: {SAVE_DIR}")

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
    model: str = DEFAULT_FORGE_MODEL
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
    sd_model_checkpoint: Optional[str] = DEFAULT_FORGE_MODEL
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
        self.current_config = {
            "sd_model_checkpoint": DEFAULT_FORGE_MODEL,
            "sd_vae": "Automatic",
            "selected_modules": {},
            "default_width": 512,
            "default_height": 512,
            "default_cfg_scale": 7.0,
            "default_steps": 20,
            "default_batch_size": 1,
            "default_batch_count": 1,
            "dynamic_prompts_enabled": False
        }
        self.config_history = []
        
    def update_config(self, new_config: Dict[str, Any]):
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
    version="1.0.0",
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

# 静的ファイル配信設定
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

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
        # Forge APIから設定を取得してVAE情報を得る
        response = requests.get(f"{FORGE_URL}/sdapi/v1/options", timeout=10)
        if response.status_code == 200:
            # 通常はsd_vae設定から利用可能なVAEを取得
            return ["Automatic", "None"]  # 基本的な選択肢
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

async def generate_image_with_forge(prompt: str, params: Dict[str, Any]):
    """Forge APIを使用して画像生成"""
    try:
        # Forge APIのtxt2imgエンドポイントを使用
        forge_params = {
            "prompt": prompt,
            "negative_prompt": params.get("negative_prompt", ""),
            "width": params.get("width", 512),
            "height": params.get("height", 512),
            "cfg_scale": params.get("cfg_scale", 7.0),
            "steps": params.get("steps", 20),
            "batch_size": params.get("batch_size", 1),
            "n_iter": params.get("batch_count", 1),
            "sampler_name": "Euler a",  # デフォルトサンプラー
            "save_images": False,  # APIでは保存しない
            "send_images": True,   # Base64で画像を返す
        }
        
        # Dynamic Prompts設定
        if params.get("dynamic_prompts", False):
            forge_params["alwayson_scripts"] = {
                "Dynamic Prompts": {
                    "args": [True]  # Dynamic Promptsを有効化
                }
            }
        
        logger.info(f"Generating image with Forge API: {forge_params}")
        response = requests.post(
            f"{FORGE_URL}/sdapi/v1/txt2img",
            json=forge_params,
            timeout=600  # 10分のタイムアウト
        )
        
        if response.status_code == 200:
            result = response.json()
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
        # Base64をデコード
        image_data = base64.b64decode(image_base64)
        
        # ファイルパス生成
        filepath = os.path.join(SAVE_DIR, filename)
        
        # 画像を保存
        with open(filepath, "wb") as f:
            f.write(image_data)
            
        logger.info(f"Image saved: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Error saving image: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving image: {str(e)}")

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
    html_content = """
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Image Generation Settings</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .form-group { margin-bottom: 15px; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input, select, textarea { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
            textarea { height: 100px; resize: vertical; }
            button { background-color: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; margin-right: 10px; }
            button:hover { background-color: #0056b3; }
            .btn-secondary { background-color: #6c757d; }
            .btn-secondary:hover { background-color: #545b62; }
            .result { margin-top: 20px; padding: 10px; background-color: #f8f9fa; border-radius: 4px; display: none; }
            .error { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
            .success { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .config-section { margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee; }
            .form-row { display: flex; gap: 15px; }
            .form-row .form-group { flex: 1; }
            .image-result { margin-top: 20px; }
            .image-result img { max-width: 100%; height: auto; border-radius: 4px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>AI Image Generation Settings</h1>
            
            <!-- 画像生成フォーム -->
            <form id="imageForm">
                <div class="form-group">
                    <label for="japanesePrompt">日本語プロンプト:</label>
                    <textarea id="japanesePrompt" name="japanese_prompt" placeholder="生成したい画像の説明を日本語で入力してください" required></textarea>
                </div>
                
                <div class="form-group">
                    <label for="negativePrompt">ネガティブプロンプト:</label>
                    <textarea id="negativePrompt" name="negative_prompt" placeholder="避けたい要素を入力"></textarea>
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label for="width">Width:</label>
                        <input type="number" id="width" name="width" value="512" min="64" max="2048" step="64">
                    </div>
                    <div class="form-group">
                        <label for="height">Height:</label>
                        <input type="number" id="height" name="height" value="512" min="64" max="2048" step="64">
                    </div>
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label for="cfgScale">CFG Scale:</label>
                        <input type="number" id="cfgScale" name="cfg_scale" value="7.0" min="1" max="30" step="0.5">
                    </div>
                    <div class="form-group">
                        <label for="steps">Steps:</label>
                        <input type="number" id="steps" name="steps" value="20" min="1" max="100">
                    </div>
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label for="batchSize">Batch Size:</label>
                        <input type="number" id="batchSize" name="batch_size" value="1" min="1" max="8">
                    </div>
                    <div class="form-group">
                        <label for="batchCount">Batch Count:</label>
                        <input type="number" id="batchCount" name="batch_count" value="1" min="1" max="10">
                    </div>
                </div>
                
                <div class="form-group">
                    <label>
                        <input type="checkbox" id="dynamicPrompts" name="dynamic_prompts"> Dynamic Prompts ON
                    </label>
                </div>
                
                <button type="submit">Generate Image</button>
                <button type="button" class="btn-secondary" onclick="loadCurrentConfig()">Load Current Settings</button>
            </form>
            
            <div id="result" class="result"></div>
            <div id="imageResult" class="image-result"></div>
            
            <!-- 設定管理セクション -->
            <div class="config-section">
                <h2>Model Configuration</h2>
                <form id="configForm">
                    <div class="form-group">
                        <label for="selectedModel">Model:</label>
                        <select id="selectedModel" name="sd_model_checkpoint">
                            <option value="">Loading...</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="selectedVae">VAE:</label>
                        <select id="selectedVae" name="sd_vae">
                            <option value="Automatic">Automatic</option>
                            <option value="None">None</option>
                        </select>
                    </div>
                    
                    <button type="button" onclick="updateConfig()">Update Configuration</button>
                    <button type="button" class="btn-secondary" onclick="loadConfigHistory()">Show History</button>
                </form>
            </div>
        </div>
        
        <script>
            // 画像生成フォーム送信
            document.getElementById('imageForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const formData = new FormData(e.target);
                const data = Object.fromEntries(formData.entries());
                
                // チェックボックスの処理
                data.dynamic_prompts = document.getElementById('dynamicPrompts').checked;
                
                // 数値型に変換
                ['width', 'height', 'steps', 'batch_size', 'batch_count'].forEach(key => {
                    data[key] = parseInt(data[key]);
                });
                data.cfg_scale = parseFloat(data.cfg_scale);
                
                showResult('Generating image...', 'success');
                
                try {
                    const response = await fetch('/get_image', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(data)
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        showResult('Image generated successfully!', 'success');
                        if (result.images && result.images.length > 0) {
                            displayImages(result.images);
                        }
                    } else {
                        showResult(`Error: ${result.detail}`, 'error');
                    }
                } catch (error) {
                    showResult(`Network error: ${error.message}`, 'error');
                }
            });
            
            // 結果表示
            function showResult(message, type) {
                const resultDiv = document.getElementById('result');
                resultDiv.textContent = message;
                resultDiv.className = `result ${type}`;
                resultDiv.style.display = 'block';
            }
            
            // 画像表示
            function displayImages(images) {
                const imageDiv = document.getElementById('imageResult');
                imageDiv.innerHTML = '';
                
                images.forEach((imageBase64, index) => {
                    const img = document.createElement('img');
                    img.src = `data:image/png;base64,${imageBase64}`;
                    img.alt = `Generated image ${index + 1}`;
                    imageDiv.appendChild(img);
                });
            }
            
            // 設定更新
            async function updateConfig() {
                const formData = new FormData(document.getElementById('configForm'));
                const data = Object.fromEntries(formData.entries());
                
                try {
                    const response = await fetch('/config', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(data)
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        showResult('Configuration updated successfully!', 'success');
                    } else {
                        showResult(`Error: ${result.detail}`, 'error');
                    }
                } catch (error) {
                    showResult(`Network error: ${error.message}`, 'error');
                }
            }
            
            // 現在の設定読み込み
            async function loadCurrentConfig() {
                try {
                    const response = await fetch('/config');
                    const result = await response.json();
                    
                    if (response.ok) {
                        // フォームに設定値を反映
                        const config = result.current_config;
                        document.getElementById('width').value = config.default_width || 512;
                        document.getElementById('height').value = config.default_height || 512;
                        document.getElementById('cfgScale').value = config.default_cfg_scale || 7.0;
                        document.getElementById('steps').value = config.default_steps || 20;
                        document.getElementById('batchSize').value = config.default_batch_size || 1;
                        document.getElementById('batchCount').value = config.default_batch_count || 1;
                        document.getElementById('dynamicPrompts').checked = config.dynamic_prompts_enabled || false;
                        
                        showResult('Current settings loaded!', 'success');
                    }
                } catch (error) {
                    showResult(`Error loading config: ${error.message}`, 'error');
                }
            }
            
            // 設定履歴表示
            async function loadConfigHistory() {
                try {
                    const response = await fetch('/config');
                    const result = await response.json();
                    
                    if (response.ok && result.config_history.length > 0) {
                        const historyText = result.config_history
                            .map(h => `${h.timestamp}: ${JSON.stringify(h.config, null, 2)}`)
                            .join('\\n\\n');
                        alert(`Configuration History:\\n\\n${historyText}`);
                    } else {
                        showResult('No configuration history available.', 'success');
                    }
                } catch (error) {
                    showResult(`Error loading history: ${error.message}`, 'error');
                }
            }
            
            // ページ読み込み時にモデル一覧を取得
            window.addEventListener('load', async function() {
                try {
                    const response = await fetch('/config');
                    const result = await response.json();
                    
                    if (response.ok) {
                        // モデル選択肢を更新
                        const modelSelect = document.getElementById('selectedModel');
                        modelSelect.innerHTML = '<option value="">Select Model</option>';
                        result.available_models.forEach(model => {
                            const option = document.createElement('option');
                            option.value = model;
                            option.textContent = model;
                            modelSelect.appendChild(option);
                        });
                        
                        // 現在の設定を反映
                        if (result.current_config.sd_model_checkpoint) {
                            modelSelect.value = result.current_config.sd_model_checkpoint;
                        }
                    }
                } catch (error) {
                    console.error('Error loading initial config:', error);
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

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
        # 1. 日本語プロンプトを英語に翻訳
        logger.info(f"Starting image generation for: {request.japanese_prompt[:50]}...")
        translated_prompt = await translate_japanese_prompt(request.japanese_prompt, request.model)
        
        # 2. 現在の設定を取得して画像生成パラメータを構築
        current_config = config_manager.get_config()
        
        # 設定値の優先順位: リクエスト > 現在の設定 > デフォルト
        params = {
            "negative_prompt": request.negative_prompt,
            "width": request.width or current_config.get("default_width", 512),
            "height": request.height or current_config.get("default_height", 512),
            "cfg_scale": request.cfg_scale or current_config.get("default_cfg_scale", 7.0),
            "steps": request.steps or current_config.get("default_steps", 20),
            "batch_size": request.batch_size or current_config.get("default_batch_size", 1),
            "batch_count": request.batch_count or current_config.get("default_batch_count", 1),
            "dynamic_prompts": request.dynamic_prompts or current_config.get("dynamic_prompts_enabled", False),
        }
        
        # 3. Forge APIで画像生成
        forge_response = await generate_image_with_forge(translated_prompt, params)
        
        # 4. 生成された画像をNASに保存
        saved_files = []
        if "images" in forge_response and forge_response["images"]:
            for i, image_base64 in enumerate(forge_response["images"]):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"generated_{timestamp}_{i:03d}.png"
                filepath = await save_image_to_nas(image_base64, filename)
                saved_files.append(filepath)
        
        # 5. レスポンスを返す（Base64のまま）
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
            # 簡単なテスト生成
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
    logger.info(f"Ollama: {OLLAMA_URL}")
    logger.info(f"Forge: {FORGE_URL}")
    logger.info(f"Save directory: {SAVE_DIR}")
    
    uvicorn.run(
        app, 
        host=TRANSLATE_HOST, 
        port=TRANSLATE_PORT,
        log_level="info"
    )