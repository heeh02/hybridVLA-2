"""Image augmentations for HybridVLA v2 training.

Reference: OpenPI transforms.py — random crop, rotation, color jitter.
Applied to PIL Images BEFORE Qwen2-VL processor tokenization.
"""

from __future__ import annotations

from typing import Optional

from PIL import Image

from vla_hybrid_v2.config import AugmentationConfig

try:
    import torchvision.transforms.v2 as T

    _HAS_TV = True
except ImportError:
    try:
        import torchvision.transforms as T  # type: ignore[no-redef]

        _HAS_TV = True
    except ImportError:
        _HAS_TV = False


class RobotImageAugmentation:
    """Training-time image augmentation for robot manipulation.

    Operates on PIL Images. Designed to be called before the VLM processor
    so that augmented pixels are correctly tokenised.
    """

    def __init__(self, cfg: AugmentationConfig) -> None:
        if not cfg.enable or not _HAS_TV:
            self.transform = None
            return

        transforms = []
        if cfg.random_crop_scale < 1.0:
            # Use 448×448 to match the target size in hdf5_adapter image resize
            transforms.append(T.RandomResizedCrop(
                size=(448, 448),
                scale=(cfg.random_crop_scale, 1.0),
                ratio=(0.95, 1.05),
                interpolation=T.InterpolationMode.BILINEAR,
                antialias=True,
            ))
        if cfg.random_rotation > 0:
            transforms.append(T.RandomRotation(degrees=cfg.random_rotation))
        if cfg.color_jitter:
            transforms.append(T.ColorJitter(
                brightness=cfg.brightness,
                contrast=cfg.contrast,
                saturation=cfg.saturation,
                hue=cfg.hue,
            ))
        self.transform = T.Compose(transforms) if transforms else None

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.transform is None:
            return image
        return self.transform(image)
