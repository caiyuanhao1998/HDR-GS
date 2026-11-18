"""HDR image I/O and temporally stable photographic tone mapping.

The global extended Reinhard operator is implemented with NumPy so one set
of scene statistics can be reused for an entire video. OpenCV handles EXR
and preview image I/O. Inputs and raw EXRs use linear RGB, not BGR.
"""
import json
import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2
import numpy as np

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def as_rgb_float(image):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3 or not image.size:
        raise ValueError("Expected a nonempty H x W x 3 RGB image")
    if not np.isfinite(image).all():
        raise ValueError("HDR image contains NaN or infinite values")
    return image


def read_hdr(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Cannot read HDR image: {}".format(path))
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    else:
        image = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2RGB)
    return as_rgb_float(image)


def write_exr(path, image):
    image = as_rgb_float(image)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    parameters = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT,
                  cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_ZIP]
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), parameters):
        raise OSError("Could not write HDR image: {}".format(path))


def fit_reinhard(images, key=0.18, white_percentile=99.95, saturation=1.0, stride=8):
    """Estimate one exposure/white point from a whole image or image sequence."""
    if not np.isfinite(key) or key <= 0:
        raise ValueError("key must be positive")
    if not 50 <= white_percentile <= 100:
        raise ValueError("white_percentile must be between 50 and 100")
    if not np.isfinite(saturation) or saturation < 0:
        raise ValueError("saturation must be nonnegative")
    if not isinstance(stride, int) or stride < 1:
        raise ValueError("stride must be a positive integer")
    samples = []
    count = 0
    for image in images:
        rgb = np.maximum(as_rgb_float(image)[::stride, ::stride], 0)
        luminance = rgb @ LUMA
        positive = luminance[luminance > 1e-8]
        if positive.size:
            samples.append(positive.astype(np.float64))
        count += 1
    if not count:
        raise ValueError("At least one HDR image is required")
    if samples:
        values = np.concatenate(samples)
        average = float(np.exp(np.log(values).mean()))
        white = max(float(np.percentile(values, white_percentile)), average)
    else:
        average = white = 1.0
    return dict(operator="extended-reinhard", log_average=average,
                white_luminance=white, key=float(key), saturation=float(saturation),
                white_percentile=float(white_percentile), calibration_images=count,
                transfer="sRGB", input="linear RGB")


def tonemap_hdr(image, settings=None, exposure=0.0):
    """Return an sRGB uint8 preview; fixed settings never normalize per frame."""
    rgb = np.maximum(as_rgb_float(image), 0).astype(np.float64)
    if not np.isfinite(exposure) or abs(exposure) > 32:
        raise ValueError("exposure must be finite and between -32 and 32 EV")
    if settings is None:
        settings = fit_reinhard([rgb])
    average = float(settings["log_average"])
    key = float(settings["key"])
    white = float(settings["white_luminance"])
    saturation = float(settings["saturation"])
    if not all(np.isfinite(x) and x > 0 for x in [average, key, white]):
        raise ValueError("Invalid tone-mapping calibration")
    if not np.isfinite(saturation) or saturation < 0:
        raise ValueError("Invalid saturation")
    luminance = rgb @ LUMA.astype(np.float64)
    base_scale = key / average
    scaled = luminance * base_scale * (2.0 ** exposure)
    # Keep the display white fixed when the user changes exposure.
    white_scaled = max(white * base_scale, 1e-8)
    display_luma = scaled * (1.0 + scaled / (white_scaled ** 2)) / (1.0 + scaled)
    chroma = rgb / np.maximum(luminance[..., None], 1e-8)
    linear = np.maximum(chroma, 0) ** saturation * display_luma[..., None]
    linear = np.clip(linear, 0.0, 1.0)
    srgb = np.where(linear <= 0.0031308, 12.92 * linear,
                    1.055 * np.power(linear, 1.0 / 2.4) - 0.055)
    return np.round(np.clip(srgb, 0, 1) * 255).astype(np.uint8)


def save_preview(path, image, settings=None, exposure=0.0):
    preview = tonemap_hdr(image, settings=settings, exposure=exposure)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)):
        raise OSError("Could not write preview: {}".format(path))
    return preview


def load_settings(path):
    return json.loads(Path(path).read_text())
