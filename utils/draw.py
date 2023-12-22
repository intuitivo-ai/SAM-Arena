# code kudos https://huggingface.co/spaces/SkalskiP/EfficientSAM
# fastSAM ultralytics
from typing import Tuple

import cv2
import numpy as np
import torch
import supervision as sv
import streamlit as st

MASK_COLOR = sv.Color.from_hex("#FF0000")
PROMPT_COLOR = sv.Color.from_hex("#D3D3D3")
MASK_ANNOTATOR = sv.MaskAnnotator(
    color=MASK_COLOR,
    color_lookup=sv.ColorLookup.INDEX)

@st.cache_data
def draw_SAM_mask_point(
    image: np.ndarray,
    detections: sv.Detections,
    x: int,
    y: int
) -> np.ndarray:
    h, w, _ = image.shape
    bgr_image = image[:, :, ::-1]
    annotated_bgr_image = MASK_ANNOTATOR.annotate(
        scene=bgr_image, detections=detections)
    annotated_bgr_image = draw_circle(
        scene=annotated_bgr_image,
        center=sv.Point(x=x, y=y),
        radius=calculate_dynamic_circle_radius(resolution_wh=(w, h)),
        color=PROMPT_COLOR)
    return annotated_bgr_image[:, :, ::-1]

def draw_circle(
    scene: np.ndarray, center: sv.Point, color: sv.Color, radius: int = 2
) -> np.ndarray:
    cv2.circle(
        scene,
        center=center.as_xy_int_tuple(),
        radius=radius,
        color=color.as_bgr(),
        thickness=-1,
    )
    return scene


def calculate_dynamic_circle_radius(resolution_wh: Tuple[int, int]) -> int:
    min_dimension = min(resolution_wh)
    if min_dimension < 480:
        return 4
    if min_dimension < 720:
        return 8
    if min_dimension < 1080:
        return 8
    if min_dimension < 2160:
        return 16
    else:
        return 16
    
def apply_masks_and_draw(image, masks, random_color=False, retinamask=True, original_h=None, original_w=None):
    """
    Applies mask annotations to the image and returns the result.

    Args:
        image (numpy.ndarray): Original image in RGB format.
        masks (numpy.ndarray): Array of mask annotations.
        random_color (bool, optional): Whether to use random color for masks. Defaults to False.
        retinamask (bool, optional): Whether to use retina mask for resizing. Defaults to True.
        original_h (int, optional): Original height of the image.
        original_w (int, optional): Original width of the image.

    Returns:
        numpy.ndarray: Image with masks applied.
    """
    if original_h is None:
        original_h = image.shape[0]
    if original_w is None:
        original_w = image.shape[1]

    n, h, w = masks.shape  # number of masks, height, width

    # Sort masks by area
    areas = np.sum(masks, axis=(1, 2))
    masks = masks[np.argsort(areas)]

    # Create mask image
    index = (masks != 0).argmax(axis=0)
    if random_color:
        color = np.random.random((n, 1, 1, 3))
    else:
        color = np.ones((n, 1, 1, 3)) * np.array([30 / 255, 144 / 255, 1.0])
    transparency = np.ones((n, 1, 1, 1)) * 0.6
    visual = np.concatenate([color, transparency], axis=-1)
    mask_image = np.expand_dims(masks, -1) * visual

    # Prepare the final image
    show = np.zeros((h, w, 4))
    h_indices, w_indices = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    indices = (index[h_indices, w_indices], h_indices, w_indices, slice(None))
    show[h_indices, w_indices, :] = mask_image[indices]

    if not retinamask:
        show = cv2.resize(show, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    # Add masks to the original image
    output_image = image.copy()
    for i in range(show.shape[2] - 1):  # Exclude the alpha channel
        output_image[:, :, i] = output_image[:, :, i] * (1 - show[:, :, 3]) + show[:, :, i] * show[:, :, 3]

    return output_image.astype(np.uint8)

def draw_FastSAM_point(detections):
    for ann in detections:
        image = ann.orig_img[..., ::-1]  # Convert BGR to RGB
        original_h, original_w = ann.orig_shape

        if ann.masks is not None:
            masks = ann.masks.data
            if isinstance(masks[0], torch.Tensor):
                masks = np.array(masks.cpu())

            output_image = apply_masks_and_draw(image, masks, random_color=True, retinamask=False, original_h=original_h, original_w=original_w)
        cv2.imwrite('output.png', output_image)
        return cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    
def draw_EdgeSAM_point(image, masks):
    # convert BGR to RGB numpy image
    image = image[..., ::-1]  # Convert BGR to RGB
    # shapes
    original_h, original_w = image.shape[:2]

    if masks is not None:
        if isinstance(masks[0], torch.Tensor):
            masks = np.array(masks.cpu())

        output_image = apply_masks_and_draw(image, masks, random_color=True, retinamask=False, original_h=original_h, original_w=original_w)
    cv2.imwrite('output.png', output_image)
    return cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)