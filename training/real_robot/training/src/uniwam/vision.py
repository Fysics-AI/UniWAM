from __future__ import annotations

import torch
import torchvision.transforms.functional as transforms_F


MATCHED_MOSAIC_HEIGHT = 320
MATCHED_MOSAIC_WIDTH = 384
MATCHED_MOSAIC_MAIN_HEIGHT = 216
MATCHED_MOSAIC_MAIN_WIDTH = 384
MATCHED_MOSAIC_WRIST_HEIGHT = 108
MATCHED_MOSAIC_WRIST_WIDTH = 192
MATCHED_MOSAIC_SOURCE_ASPECT = 424.0 / 240.0
ROBOTWIN_MOSAIC_HEIGHT = 384
ROBOTWIN_MOSAIC_WIDTH = 320


def _validate_camera_tensor(name: str, image: torch.Tensor) -> None:
    if image.ndim < 3 or image.shape[-3] != 3:
        raise ValueError(
            f"{name} must have shape [...,3,H,W], got {tuple(image.shape)}."
        )
    height, width = int(image.shape[-2]), int(image.shape[-1])
    if height <= 0 or width <= 0:
        raise ValueError(f"{name} has invalid spatial shape {height}x{width}.")
    aspect = float(width) / float(height)
    relative_error = abs(aspect / MATCHED_MOSAIC_SOURCE_ASPECT - 1.0)
    if relative_error > 0.02:
        raise ValueError(
            f"{name} aspect ratio {aspect:.5f} differs from the expected "
            f"424x240 ratio by {relative_error:.2%}; refusing to distort it."
        )


def build_matched_fastwam_mosaic(
    main: torch.Tensor,
    left_wrist: torch.Tensor,
    right_wrist: torch.Tensor,
    *,
    fill: float = 0.5,
) -> torch.Tensor:
    """Build a 384x320 canvas via a 384x324 16:9 mosaic and 2px center crop."""
    del fill
    for name, image in (
        ("main", main),
        ("left_wrist", left_wrist),
        ("right_wrist", right_wrist),
    ):
        _validate_camera_tensor(name, image)
    if main.shape[:-3] != left_wrist.shape[:-3] or main.shape[:-3] != right_wrist.shape[:-3]:
        raise ValueError(
            "Camera leading dimensions must match, got "
            f"main={tuple(main.shape)}, left={tuple(left_wrist.shape)}, "
            f"right={tuple(right_wrist.shape)}."
        )

    resize_kwargs = {
        "interpolation": transforms_F.InterpolationMode.BILINEAR,
        "antialias": True,
    }
    main = transforms_F.resize(
        main,
        size=[MATCHED_MOSAIC_MAIN_HEIGHT, MATCHED_MOSAIC_MAIN_WIDTH],
        **resize_kwargs,
    )
    left_wrist = transforms_F.resize(
        left_wrist,
        size=[MATCHED_MOSAIC_WRIST_HEIGHT, MATCHED_MOSAIC_WRIST_WIDTH],
        **resize_kwargs,
    )
    right_wrist = transforms_F.resize(
        right_wrist,
        size=[MATCHED_MOSAIC_WRIST_HEIGHT, MATCHED_MOSAIC_WRIST_WIDTH],
        **resize_kwargs,
    )

    bottom = torch.cat([left_wrist, right_wrist], dim=-1)
    mosaic = torch.cat([main, bottom], dim=-2)
    if mosaic.shape[-2:] != (324, 384):
        raise AssertionError(f"Unexpected pre-crop mosaic shape: {tuple(mosaic.shape)}")
    mosaic = mosaic[..., 2:322, :]
    if mosaic.shape[-2:] != (MATCHED_MOSAIC_HEIGHT, MATCHED_MOSAIC_WIDTH):
        raise AssertionError(f"Unexpected matched mosaic shape: {tuple(mosaic.shape)}")
    return mosaic


def _resize_aspect_fit_edge_pad(
    image: torch.Tensor,
    *,
    target_height: int,
    target_width: int,
) -> torch.Tensor:
    height, width = int(image.shape[-2]), int(image.shape[-1])
    scale = min(float(target_height) / height, float(target_width) / width)
    resized_height = max(1, min(target_height, int(round(height * scale))))
    resized_width = max(1, min(target_width, int(round(width * scale))))
    image = transforms_F.resize(
        image,
        size=[resized_height, resized_width],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    pad_h = target_height - resized_height
    pad_w = target_width - resized_width
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return transforms_F.pad(image, [left, top, right, bottom], padding_mode="edge")


def build_aspect_padded_fastwam_mosaic(
    main: torch.Tensor,
    left_wrist: torch.Tensor,
    right_wrist: torch.Tensor,
) -> torch.Tensor:
    """Preserve arbitrary camera geometry before applying the matched mosaic."""
    target = {"target_height": 240, "target_width": 424}
    return build_matched_fastwam_mosaic(
        _resize_aspect_fit_edge_pad(main, **target),
        _resize_aspect_fit_edge_pad(left_wrist, **target),
        _resize_aspect_fit_edge_pad(right_wrist, **target),
    )


def build_nav_high_fastwam_mosaic(
    nav: torch.Tensor,
    high: torch.Tensor,
) -> torch.Tensor:
    """Build the legacy 384x320 navigation canvas: D455 over D435 high."""
    _validate_camera_tensor("nav", nav)
    _validate_camera_tensor("high", high)
    if nav.shape[:-3] != high.shape[:-3]:
        raise ValueError(
            f"Camera leading dimensions must match, got nav={tuple(nav.shape)}, high={tuple(high.shape)}."
        )
    nav = _resize_aspect_fit_edge_pad(nav, target_height=192, target_width=320)
    high = _resize_aspect_fit_edge_pad(high, target_height=192, target_width=320)
    mosaic = torch.cat([nav, high], dim=-2)
    if mosaic.shape[-2:] != (ROBOTWIN_MOSAIC_HEIGHT, ROBOTWIN_MOSAIC_WIDTH):
        raise AssertionError(f"Unexpected nav/high mosaic shape: {tuple(mosaic.shape)}")
    return mosaic


def build_robotwin_fastwam_mosaic(
    main: torch.Tensor,
    left_wrist: torch.Tensor,
    right_wrist: torch.Tensor,
) -> torch.Tensor:
    """Build the legacy 384x320 manipulation canvas used by FastWAM."""
    for name, image in (
        ("main", main),
        ("left_wrist", left_wrist),
        ("right_wrist", right_wrist),
    ):
        _validate_camera_tensor(name, image)
    if main.shape[:-3] != left_wrist.shape[:-3] or main.shape[:-3] != right_wrist.shape[:-3]:
        raise ValueError(
            "Camera leading dimensions must match, got "
            f"main={tuple(main.shape)}, left={tuple(left_wrist.shape)}, right={tuple(right_wrist.shape)}."
        )
    resize_kwargs = {
        "interpolation": transforms_F.InterpolationMode.BILINEAR,
        "antialias": True,
    }
    main = transforms_F.resize(main, size=[256, 320], **resize_kwargs)
    left_wrist = transforms_F.resize(left_wrist, size=[128, 160], **resize_kwargs)
    right_wrist = transforms_F.resize(right_wrist, size=[128, 160], **resize_kwargs)
    mosaic = torch.cat([main, torch.cat([left_wrist, right_wrist], dim=-1)], dim=-2)
    if mosaic.shape[-2:] != (ROBOTWIN_MOSAIC_HEIGHT, ROBOTWIN_MOSAIC_WIDTH):
        raise AssertionError(f"Unexpected Robotwin mosaic shape: {tuple(mosaic.shape)}")
    return mosaic
