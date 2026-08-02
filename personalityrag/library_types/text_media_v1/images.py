from __future__ import annotations

import hashlib
import io
import os
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError


MAX_SOURCE_BYTES = 25 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
MAX_LONG_EDGE = 1536
MAX_CANONICAL_BYTES = 1536 * 1024
PREVIEW_EDGE = 384
ALLOWED_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})


class ImageValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class NormalizedImage:
    data: bytes
    sha256: str
    width: int
    height: int
    size_bytes: int
    mime_type: str = "image/webp"


def _to_srgb(image: Image.Image) -> Image.Image:
    icc = image.info.get("icc_profile")
    if icc:
        try:
            source = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            target = ImageCms.createProfile("sRGB")
            mode = "RGBA" if "A" in image.getbands() else "RGB"
            return ImageCms.profileToProfile(image, source, target, outputMode=mode)
        except (ImageCms.PyCMSError, OSError, ValueError):
            pass
    return image.convert("RGBA" if "A" in image.getbands() else "RGB")


def _encode_webp(image: Image.Image) -> bytes:
    working = image
    for _ in range(8):
        for quality in (80, 75, 70, 65, 60):
            output = io.BytesIO()
            working.save(
                output,
                format="WEBP",
                quality=quality,
                method=6,
                exact=True,
            )
            data = output.getvalue()
            if len(data) <= MAX_CANONICAL_BYTES:
                return data
        next_size = (
            max(256, int(working.width * 0.88)),
            max(256, int(working.height * 0.88)),
        )
        if next_size == working.size:
            break
        working = working.resize(next_size, Image.Resampling.LANCZOS)
    raise ImageValidationError("图片压缩后仍超过 1.5 MiB")


def normalize_image_bytes(data: bytes) -> NormalizedImage:
    if not data or len(data) > MAX_SOURCE_BYTES:
        raise ImageValidationError("图片为空或超过 25 MiB")
    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.format not in ALLOWED_FORMATS:
                raise ImageValidationError("仅支持 PNG、JPEG、WebP 图片")
            if getattr(source, "n_frames", 1) != 1:
                raise ImageValidationError("不支持动画图片")
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise ImageValidationError("图片像素总量超过限制")
            source.load()
            image = ImageOps.exif_transpose(source)
            image = _to_srgb(image)
            if max(image.size) > MAX_LONG_EDGE:
                image.thumbnail((MAX_LONG_EDGE, MAX_LONG_EDGE), Image.Resampling.LANCZOS)
            canonical = _encode_webp(image)
            with Image.open(io.BytesIO(canonical)) as verified:
                verified.verify()
            digest = hashlib.sha256(canonical).hexdigest()
            return NormalizedImage(
                data=canonical,
                sha256=digest,
                width=image.width,
                height=image.height,
                size_bytes=len(canonical),
            )
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ImageValidationError("图片文件无效或像素尺寸不安全") from exc
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit


def write_content_addressed_image(root: Path, normalized: NormalizedImage) -> Path:
    relative = Path("assets") / "images" / "sha256" / normalized.sha256[:2] / f"{normalized.sha256}.webp"
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(normalized.data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    return relative


def build_preview(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.thumbnail((PREVIEW_EDGE, PREVIEW_EDGE), Image.Resampling.LANCZOS)
        temporary = target.with_suffix(".tmp")
        image.save(temporary, format="WEBP", quality=72, method=6)
        os.replace(temporary, target)
