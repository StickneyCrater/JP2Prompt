import os
import requests
import logging
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Forge API設定
FORGE_HOST = os.getenv("FORGE_HOST", "192.168.2.197")
FORGE_PORT = os.getenv("FORGE_PORT", "7865")
FORGE_URL = f"http://{FORGE_HOST}:{FORGE_PORT}"

# Forge APIのProxy用ルーター
forge_router = APIRouter(prefix="/sd", tags=["forge_proxy"])

@forge_router.post("/sdapi/v1/txt2img", 
                   summary="Text to Image Generation",
                   description="Generate images from text prompts using Stable Diffusion. This endpoint accepts detailed generation parameters and returns base64-encoded images.")
async def proxy_txt2img(request: Dict[str, Any]):
    """
    Generate images from text prompts using Stable Diffusion.
    
    This endpoint proxies requests to the Automatic1111 Forge API for text-to-image generation.
    All parameters are passed through without modification.
    
    Parameters:
    - prompt: The text prompt describing the desired image
    - negative_prompt: Text describing what to avoid in the image
    - width: Image width in pixels (default: 512)
    - height: Image height in pixels (default: 512)
    - cfg_scale: Classifier Free Guidance scale (default: 7.0)
    - steps: Number of sampling steps (default: 20)
    - batch_size: Number of images to generate per batch
    - sampler_name: Sampling method to use
    - And many other parameters supported by Forge API
    
    Returns:
    - images: List of base64-encoded generated images
    - parameters: The parameters used for generation
    - info: Generation metadata and settings
    """
    try:
        response = requests.post(
            f"{FORGE_URL}/sdapi/v1/txt2img",
            json=request,
            timeout=600
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Forge API error: {response.text}"
            )
            
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Request timeout")
    except Exception as e:
        logger.error(f"Proxy txt2img error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")

@forge_router.post("/sdapi/v1/img2img",
                   summary="Image to Image Generation", 
                   description="Generate images from existing images with text prompts. Useful for image variations, inpainting, and style transfer.")
async def proxy_img2img(request: Dict[str, Any]):
    """
    Generate images from existing images with text prompts.
    
    This endpoint proxies requests to the Automatic1111 Forge API for image-to-image generation.
    
    Parameters include all txt2img parameters plus:
    - init_images: List of base64-encoded input images
    - denoising_strength: How much to change the input image (0.0-1.0)
    - resize_mode: How to handle input image resizing
    - mask: Optional mask for inpainting (base64-encoded)
    
    Returns same format as txt2img endpoint.
    """
    try:
        response = requests.post(
            f"{FORGE_URL}/sdapi/v1/img2img",
            json=request,
            timeout=600
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Forge API error: {response.text}"
            )
            
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Request timeout")
    except Exception as e:
        logger.error(f"Proxy img2img error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")

@forge_router.get("/sdapi/v1/options",
                  summary="Get Current Options",
                  description="Retrieve current Forge configuration settings including model, VAE, and other generation parameters.")
async def proxy_get_options():
    """Get current Forge configuration settings."""
    try:
        response = requests.get(f"{FORGE_URL}/sdapi/v1/options", timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Forge API error: {response.text}"
            )
            
    except Exception as e:
        logger.error(f"Proxy get options error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")

@forge_router.post("/sdapi/v1/options",
                   summary="Update Configuration Options",
                   description="Update Forge configuration settings such as model checkpoint, VAE, sampling settings, etc.")
async def proxy_set_options(request: Dict[str, Any]):
    """Update Forge configuration settings."""
    try:
        response = requests.post(
            f"{FORGE_URL}/sdapi/v1/options",
            json=request,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Forge API error: {response.text}"
            )
            
    except Exception as e:
        logger.error(f"Proxy set options error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")

@forge_router.get("/sdapi/v1/sd-models",
                  summary="List Available Models",
                  description="Get list of all available Stable Diffusion model checkpoints that can be loaded.")
async def proxy_get_models():
    """Get list of available Stable Diffusion models."""
    try:
        response = requests.get(f"{FORGE_URL}/sdapi/v1/sd-models", timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Forge API error: {response.text}"
            )
            
    except Exception as e:
        logger.error(f"Proxy get models error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")

@forge_router.get("/sdapi/v1/sd-modules",
                  summary="List Available Modules",
                  description="Get list of available SD modules including VAEs, text encoders, and UNETs. Essential for Flux model configuration.")
async def proxy_get_modules():
    """Get list of available SD modules (VAEs, text encoders, UNETs)."""
    try:
        response = requests.get(f"{FORGE_URL}/sdapi/v1/sd-modules", timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Forge API error: {response.text}"
            )
            
    except Exception as e:
        logger.error(f"Proxy get modules error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")

@forge_router.get("/sdapi/v1/samplers",
                  summary="List Available Samplers",
                  description="Get list of available sampling methods/algorithms for image generation.")
async def proxy_get_samplers():
    """Get list of available samplers."""
    try:
        response = requests.get(f"{FORGE_URL}/sdapi/v1/samplers", timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Forge API error: {response.text}"
            )
            
    except Exception as e:
        logger.error(f"Proxy get samplers error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")

@forge_router.get("/sdapi/v1/schedulers",
                  summary="List Available Schedulers",
                  description="Get list of available noise schedulers for sampling process.")
async def proxy_get_schedulers():
    """Get list of available schedulers."""
    try:
        response = requests.get(f"{FORGE_URL}/sdapi/v1/schedulers", timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Forge API error: {response.text}"
            )
            
    except Exception as e:
        logger.error(f"Proxy get schedulers error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")

@forge_router.get("/sdapi/v1/upscalers",
                  summary="List Available Upscalers",
                  description="Get list of available upscaling models for image enhancement.")
async def proxy_get_upscalers():
    """Get list of available upscalers."""
    try:
        response = requests.get(f"{FORGE_URL}/sdapi/v1/upscalers", timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Forge API error: {response.text}"
            )
            
    except Exception as e:
        logger.error(f"Proxy get upscalers error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")

@forge_router.get("/sdapi/v1/progress",
                  summary="Get Generation Progress", 
                  description="Get current generation progress and status. Useful for monitoring long-running generation tasks.")
async def proxy_get_progress(skip_current_image: bool = False):
    """Get current generation progress."""
    try:
        params = {"skip_current_image": skip_current_image}
        response = requests.get(f"{FORGE_URL}/sdapi/v1/progress", params=params, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Forge API error: {response.text}"
            )
            
    except Exception as e:
        logger.error(f"Proxy get progress error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")

@forge_router.post("/sdapi/v1/interrupt",
                   summary="Interrupt Generation",
                   description="Stop the current image generation process.")
async def proxy_interrupt():
    """Interrupt current generation."""
    try:
        response = requests.post(f"{FORGE_URL}/sdapi/v1/interrupt", timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Forge API error: {response.text}"
            )
            
    except Exception as e:
        logger.error(f"Proxy interrupt error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")

@forge_router.post("/sdapi/v1/skip",
                   summary="Skip Current Generation Step",
                   description="Skip the current generation step and proceed to the next one.")
async def proxy_skip():
    """Skip current generation step."""
    try:
        response = requests.post(f"{FORGE_URL}/sdapi/v1/skip", timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Forge API error: {response.text}"
            )
            
    except Exception as e:
        logger.error(f"Proxy skip error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")

@forge_router.post("/sdapi/v1/extra-single-image",
                   summary="Upscale Single Image",
                   description="Upscale and enhance a single image using various upscaling models and face restoration.")
async def proxy_extra_single_image(request: Dict[str, Any]):
    """Upscale and enhance a single image."""
    try:
        response = requests.post(
            f"{FORGE_URL}/sdapi/v1/extra-single-image",
            json=request,
            timeout=300
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Forge API error: {response.text}"
            )
            
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Request timeout")
    except Exception as e:
        logger.error(f"Proxy extra single image error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")

@forge_router.post("/sdapi/v1/extra-batch-images",
                   summary="Upscale Batch Images",
                   description="Upscale and enhance multiple images in batch using various upscaling models.")
async def proxy_extra_batch_images(request: Dict[str, Any]):
    """Upscale and enhance multiple images in batch."""
    try:
        response = requests.post(
            f"{FORGE_URL}/sdapi/v1/extra-batch-images",
            json=request,
            timeout=600
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Forge API error: {response.text}"
            )
            
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Request timeout")
    except Exception as e:
        logger.error(f"Proxy extra batch images error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")

@forge_router.post("/sdapi/v1/png-info",
                   summary="Extract PNG Info",
                   description="Extract generation parameters and metadata from a PNG image.")
async def proxy_png_info(request: Dict[str, Any]):
    """Extract PNG generation info."""
    try:
        response = requests.post(
            f"{FORGE_URL}/sdapi/v1/png-info",
            json=request,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Forge API error: {response.text}"
            )
            
    except Exception as e:
        logger.error(f"Proxy png info error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")

@forge_router.post("/sdapi/v1/interrogate",
                   summary="Interrogate Image",
                   description="Generate text description of an image using CLIP or other interrogation models.")
async def proxy_interrogate(request: Dict[str, Any]):
    """Interrogate image to generate description."""
    try:
        response = requests.post(
            f"{FORGE_URL}/sdapi/v1/interrogate",
            json=request,
            timeout=120
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Forge API error: {response.text}"
            )
            
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Request timeout")
    except Exception as e:
        logger.error(f"Proxy interrogate error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")

@forge_router.get("/sdapi/v1/memory",
                  summary="Get Memory Usage",
                  description="Get current memory usage statistics for RAM and VRAM.")
async def proxy_get_memory():
    """Get memory usage statistics."""
    try:
        response = requests.get(f"{FORGE_URL}/sdapi/v1/memory", timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Forge API error: {response.text}"
            )
            
    except Exception as e:
        logger.error(f"Proxy get memory error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")

@forge_router.post("/sdapi/v1/refresh-checkpoints",
                   summary="Refresh Model Checkpoints",
                   description="Refresh the list of available model checkpoints from disk.")
async def proxy_refresh_checkpoints():
    """Refresh model checkpoints list."""
    try:
        response = requests.post(f"{FORGE_URL}/sdapi/v1/refresh-checkpoints", timeout=60)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Forge API error: {response.text}"
            )
            
    except Exception as e:
        logger.error(f"Proxy refresh checkpoints error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")

@forge_router.post("/sdapi/v1/refresh-vae",
                   summary="Refresh VAE Models",
                   description="Refresh the list of available VAE models from disk.")
async def proxy_refresh_vae():
    """Refresh VAE models list."""
    try:
        response = requests.post(f"{FORGE_URL}/sdapi/v1/refresh-vae", timeout=60)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Forge API error: {response.text}"
            )
            
    except Exception as e:
        logger.error(f"Proxy refresh VAE error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")

@forge_router.get("/sdapi/v1/cmd-flags",
                  summary="Get Command Line Flags",
                  description="Get the command line flags used to start Forge.")
async def proxy_get_cmd_flags():
    """Get command line flags."""
    try:
        response = requests.get(f"{FORGE_URL}/sdapi/v1/cmd-flags", timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Forge API error: {response.text}"
            )
            
    except Exception as e:
        logger.error(f"Proxy get cmd flags error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")

# ControlNet関連のプロキシエンドポイント
@forge_router.get("/controlnet/model_list",
                  summary="Get ControlNet Models",
                  description="Get list of available ControlNet models for conditional generation.")
async def proxy_controlnet_models():
    """Get ControlNet models list."""
    try:
        response = requests.get(f"{FORGE_URL}/controlnet/model_list", timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Forge API error: {response.text}"
            )
            
    except Exception as e:
        logger.error(f"Proxy ControlNet models error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")

@forge_router.get("/controlnet/module_list",
                  summary="Get ControlNet Modules",
                  description="Get list of available ControlNet preprocessor modules.")
async def proxy_controlnet_modules():
    """Get ControlNet modules list."""
    try:
        response = requests.get(f"{FORGE_URL}/controlnet/module_list", timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Forge API error: {response.text}"
            )
            
    except Exception as e:
        logger.error(f"Proxy ControlNet modules error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")

@forge_router.post("/controlnet/detect",
                   summary="ControlNet Preprocessing",
                   description="Preprocess images using ControlNet modules for conditional generation.")
async def proxy_controlnet_detect(request: Dict[str, Any]):
    """ControlNet image preprocessing."""
    try:
        response = requests.post(
            f"{FORGE_URL}/controlnet/detect",
            json=request,
            timeout=120
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Forge API error: {response.text}"
            )
            
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Request timeout")
    except Exception as e:
        logger.error(f"Proxy ControlNet detect error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")