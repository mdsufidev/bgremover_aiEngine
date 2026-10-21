import io
import re
import logging
from contextlib import asynccontextmanager
from typing import Optional, Tuple
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import numpy as np
import scipy.ndimage
from rembg import new_session, remove
import onnxruntime as ort

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ai-engine")

session_holder = {}

ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp"
}

MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024
FREE_MAX_DIMENSION = 1080  # Free tier resolution cap: 1080x1080


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_name = "u2net"
    
    # Check for NVIDIA CUDA GPU Acceleration
    available_providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in available_providers:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        logger.info("🚀 NVIDIA GPU Acceleration detected! Initializing model with CUDAExecutionProvider...")
    else:
        providers = ["CPUExecutionProvider"]
        logger.info("⚡ Initializing model with CPUExecutionProvider...")

    try:
        session_holder["session"] = new_session(model_name, providers=providers)
        logger.info(f"Model '{model_name}' successfully pre-warmed in memory.")
    except Exception as e:
        logger.error(f"Failed to load model '{model_name}': {str(e)}")
        try:
            logger.info("Attempting fallback to CPU session...")
            session_holder["session"] = new_session(model_name, providers=["CPUExecutionProvider"])
            logger.info("Fallback CPU session loaded successfully.")
        except Exception as ex2:
            logger.error(f"Fallback session failed: {str(ex2)}")
            raise ex2
    yield
    session_holder.clear()
    logger.info("Model session cleared.")


app = FastAPI(
    title="AI Background Remover Microservice",
    description="High-performance background & color removal engine with Tiered Resolution Control",
    version="1.2.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)


def apply_resolution_limit(
    pil_img: Image.Image,
    is_premium: bool,
    max_dimension: int = FREE_MAX_DIMENSION
) -> Tuple[Image.Image, bool, Tuple[int, int]]:
    """
    Preserves original full resolution for high-definition studio preview and editing.
    Resolution limits (1080x1080 Free vs Original Ultra-HD Pro) are enforced during download export.
    """
    orig_w, orig_h = pil_img.size
    max_safe = 8192
    if orig_w > max_safe or orig_h > max_safe:
        scale = min(max_safe / orig_w, max_safe / orig_h)
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))
        logger.warning(f"Safety clamp for ultra-huge image: {orig_w}x{orig_h} -> {new_w}x{new_h}")
        return pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS), True, (new_w, new_h)

    logger.info(f"Processing image at original resolution ({orig_w}x{orig_h}), premium={is_premium}")
    return pil_img, False, (orig_w, orig_h)


def parse_color(color_str: str) -> Tuple[int, int, int]:
    color_str = color_str.strip()
    if color_str.startswith("#"):
        color_str = color_str[1:]
    
    if len(color_str) == 6 and re.match(r"^[0-9a-fA-F]{6}$", color_str):
        r = int(color_str[0:2], 16)
        g = int(color_str[2:4], 16)
        b = int(color_str[4:6], 16)
        return (r, g, b)
    
    if len(color_str) == 3 and re.match(r"^[0-9a-fA-F]{3}$", color_str):
        r = int(color_str[0] * 2, 16)
        g = int(color_str[1] * 2, 16)
        b = int(color_str[2] * 2, 16)
        return (r, g, b)
    
    if "," in color_str:
        parts = [int(p.strip()) for p in color_str.split(",") if p.strip().isdigit()]
        if len(parts) >= 3:
            return (min(255, max(0, parts[0])), min(255, max(0, parts[1])), min(255, max(0, parts[2])))
            
    raise ValueError(f"Invalid color format '{color_str}'. Expected hex (#RRGGBB) or RGB (R,G,B).")


def remove_color_from_image(
    rgba_img: Image.Image,
    target_rgb: Tuple[int, int, int],
    tolerance: int = 25,
    contiguous: bool = False,
    click_x: Optional[int] = None,
    click_y: Optional[int] = None,
    feather_radius: int = 2
) -> bytes:
    arr = np.array(rgba_img, dtype=np.float32)
    h, w, _ = arr.shape
    
    rgb = arr[:, :, :3]
    target = np.array(target_rgb, dtype=np.float32)
    dist = np.linalg.norm(rgb - target, axis=2)
    
    max_dist = 441.67295593
    threshold = (max(1, min(100, tolerance)) / 100.0) * max_dist
    matches = dist <= threshold
    
    if contiguous and click_x is not None and click_y is not None:
        cx = int(np.clip(click_x, 0, w - 1))
        cy = int(np.clip(click_y, 0, h - 1))
        
        labeled, _ = scipy.ndimage.label(matches)
        seed_label = labeled[cy, cx]
        
        if seed_label > 0:
            mask = (labeled == seed_label)
        else:
            local_dist = dist[max(0, cy - 2):min(h, cy + 3), max(0, cx - 2):min(w, cx + 3)]
            seed_threshold = max(threshold, float(np.min(local_dist)) + 5.0)
            expanded_matches = dist <= seed_threshold
            labeled, _ = scipy.ndimage.label(expanded_matches)
            seed_label = labeled[cy, cx]
            mask = (labeled == seed_label) if seed_label > 0 else matches
    else:
        mask = matches
    
    alpha = arr[:, :, 3].copy()
    feather_band = max(2.0, threshold * 0.15)
    alpha[mask] = 0.0
    
    if feather_radius > 0:
        boundary = scipy.ndimage.binary_dilation(mask, iterations=feather_radius) & (~mask)
        dist_boundary = dist[boundary]
        edge_factor = np.clip((dist_boundary - threshold) / feather_band, 0.0, 1.0)
        alpha[boundary] = alpha[boundary] * edge_factor

    arr[:, :, 3] = np.clip(alpha, 0, 255)
    result_img = Image.fromarray(arr.astype(np.uint8), mode="RGBA")
    
    out_buffer = io.BytesIO()
    result_img.save(out_buffer, format="PNG")
    return out_buffer.getvalue()


@app.get("/health", status_code=status.HTTP_200_OK, tags=["Health"])
async def health_check():
    is_ready = "session" in session_holder and session_holder["session"] is not None
    available_providers = ort.get_available_providers()
    gpu_active = "CUDAExecutionProvider" in available_providers
    return {
        "status": "UP" if is_ready else "DOWN",
        "service": "ai-background-remover",
        "model_loaded": is_ready,
        "gpu_accelerated": gpu_active,
        "providers": available_providers,
        "free_max_resolution": f"{FREE_MAX_DIMENSION}x{FREE_MAX_DIMENSION}"
    }


@app.post(
    "/api/v1/remove-bg",
    response_class=Response,
    status_code=status.HTTP_200_OK,
    tags=["Background Removal"]
)
async def remove_background(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, WEBP)"),
    is_premium: bool = Form(False, description="Whether the user has premium / admin ultra-HD access")
):
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported media type '{file.content_type}'."
        )

    try:
        input_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to read file.")

    if len(input_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File exceeds maximum size of {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB."
        )

    try:
        with Image.open(io.BytesIO(input_bytes)) as pil_img:
            # Check and enforce resolution tiering
            processed_input_img, was_downscaled, final_dim = apply_resolution_limit(
                pil_img=pil_img,
                is_premium=is_premium,
                max_dimension=FREE_MAX_DIMENSION
            )
            buf = io.BytesIO()
            processed_input_img.save(buf, format="PNG")
            ready_bytes = buf.getvalue()
    except Exception as e:
        logger.error(f"Image preprocessing failed: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Corrupted or invalid image.")

    try:
        session = session_holder.get("session")
        output_bytes = remove(ready_bytes, session=session)
    except Exception as e:
        logger.error(f"Inference error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Inference failed.")

    return Response(
        content=output_bytes,
        media_type="image/png",
        headers={
            "Content-Disposition": "inline; filename=bg_removed.png",
            "Cache-Control": "no-cache",
            "X-Is-Downscaled": str(was_downscaled).lower(),
            "X-Resolution": f"{final_dim[0]}x{final_dim[1]}",
            "X-Tier": "PREMIUM" if is_premium else "FREE"
        }
    )


@app.post(
    "/api/v1/remove-color",
    response_class=Response,
    status_code=status.HTTP_200_OK,
    tags=["Color-Based Removal"]
)
async def remove_color(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, WEBP)"),
    color: str = Form(..., description="Target color in Hex (#RRGGBB) or RGB ('255,255,255')"),
    tolerance: int = Form(25, description="Color tolerance percentage from 1 to 100"),
    contiguous: bool = Form(False, description="Whether to erase only connected region"),
    x: Optional[int] = Form(None, description="Clicked pixel X coordinate"),
    y: Optional[int] = Form(None, description="Clicked pixel Y coordinate"),
    is_premium: bool = Form(False, description="Whether user has premium/admin access")
):
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported media type.")

    try:
        target_rgb = parse_color(color)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))

    try:
        input_bytes = await file.read()
        with Image.open(io.BytesIO(input_bytes)) as pil_img:
            orig_w, orig_h = pil_img.size
            rgba_img = pil_img.convert("RGBA")
            processed_img, was_downscaled, final_dim = apply_resolution_limit(rgba_img, is_premium, FREE_MAX_DIMENSION)
            
            # Re-scale coordinates if image was downscaled
            scaled_x = x
            scaled_y = y
            if was_downscaled and x is not None and y is not None:
                scale_w = final_dim[0] / orig_w
                scale_h = final_dim[1] / orig_h
                scaled_x = int(round(x * scale_w))
                scaled_y = int(round(y * scale_h))

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to parse image.")

    try:
        output_bytes = remove_color_from_image(
            rgba_img=processed_img,
            target_rgb=target_rgb,
            tolerance=tolerance,
            contiguous=contiguous,
            click_x=scaled_x,
            click_y=scaled_y
        )
    except Exception as e:
        logger.error(f"Color removal failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to erase color.")

    return Response(
        content=output_bytes,
        media_type="image/png",
        headers={
            "Content-Disposition": "inline; filename=color_removed.png",
            "Cache-Control": "no-cache",
            "X-Is-Downscaled": str(was_downscaled).lower(),
            "X-Resolution": f"{final_dim[0]}x{final_dim[1]}",
            "X-Tier": "PREMIUM" if is_premium else "FREE"
        }
    )


def enhance_and_restore_image(
    pil_img: Image.Image,
    is_premium: bool,
    strength: int = 50,
    upscale: bool = True
) -> Tuple[bytes, bool, Tuple[int, int]]:
    orig_w, orig_h = pil_img.size
    has_alpha = pil_img.mode in ("RGBA", "LA") or (pil_img.mode == "P" and "transparency" in pil_img.info)
    if has_alpha:
        rgba = pil_img.convert("RGBA")
        r, g, b, alpha = rgba.split()
        rgb_img = Image.merge("RGB", (r, g, b))
    else:
        rgb_img = pil_img.convert("RGB")
        alpha = None

    factor = max(0.1, min(1.0, strength / 100.0))

    try:
        rgb_img = ImageOps.autocontrast(rgb_img, cutoff=0.5)
    except Exception:
        pass

    color_enhancer = ImageEnhance.Color(rgb_img)
    rgb_img = color_enhancer.enhance(1.0 + (0.25 * factor))

    contrast_enhancer = ImageEnhance.Contrast(rgb_img)
    rgb_img = contrast_enhancer.enhance(1.0 + (0.2 * factor))

    mask_percent = int(100 + (100 * factor))
    rgb_img = rgb_img.filter(ImageFilter.UnsharpMask(radius=2.0, percent=mask_percent, threshold=3))

    sharp_enhancer = ImageEnhance.Sharpness(rgb_img)
    rgb_img = sharp_enhancer.enhance(1.0 + (0.5 * factor))

    if upscale:
        target_w = orig_w * 2
        target_h = orig_h * 2
        if not is_premium:
            if target_w > FREE_MAX_DIMENSION or target_h > FREE_MAX_DIMENSION:
                scale = min(FREE_MAX_DIMENSION / target_w, FREE_MAX_DIMENSION / target_h)
                target_w = max(1, int(round(target_w * scale)))
                target_h = max(1, int(round(target_h * scale)))
        rgb_img = rgb_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        if alpha is not None:
            alpha = alpha.resize((target_w, target_h), Image.Resampling.LANCZOS)
        rgb_img = rgb_img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=80, threshold=2))
        final_dim = rgb_img.size
        was_downscaled = (final_dim[0] < orig_w * 2) if not is_premium else False
    else:
        rgb_img, was_downscaled, final_dim = apply_resolution_limit(rgb_img, is_premium, FREE_MAX_DIMENSION)
        if alpha is not None and was_downscaled:
            alpha = alpha.resize(final_dim, Image.Resampling.LANCZOS)

    if alpha is not None:
        final_img = Image.merge("RGBA", (*rgb_img.split(), alpha))
        format_type = "PNG"
    else:
        final_img = rgb_img
        format_type = "PNG"

    buf = io.BytesIO()
    final_img.save(buf, format="PNG", optimize=True)
    return buf.getvalue(), was_downscaled, final_dim


@app.post(
    "/api/v1/restore-image",
    response_class=Response,
    status_code=status.HTTP_200_OK,
    tags=["Photo Restoration & Enhancement"]
)
async def restore_image(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, WEBP)"),
    strength: int = Form(50, description="Restoration & sharpness intensity 1-100"),
    upscale: bool = Form(True, description="2x Super-Resolution Lanczos upscaling"),
    is_premium: bool = Form(False, description="Whether user has premium/admin access")
):
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported media type.")

    try:
        input_bytes = await file.read()
        with Image.open(io.BytesIO(input_bytes)) as pil_img:
            output_bytes, was_downscaled, final_dim = enhance_and_restore_image(
                pil_img=pil_img,
                is_premium=is_premium,
                strength=strength,
                upscale=upscale
            )
    except Exception as e:
        logger.error(f"Image restoration failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to restore image.")

    return Response(
        content=output_bytes,
        media_type="image/png",
        headers={
            "Content-Disposition": "inline; filename=restored_image.png",
            "Cache-Control": "no-cache",
            "X-Is-Downscaled": str(was_downscaled).lower(),
            "X-Resolution": f"{final_dim[0]}x{final_dim[1]}",
            "X-Tier": "PREMIUM" if is_premium else "FREE"
        }
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
