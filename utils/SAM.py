
import torch
import numpy as np
from PIL import Image
import streamlit as st
import supervision as sv
from ultralytics import YOLO
from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt
from transformers import SamModel, SamProcessor

from utils.efficient_sam import load, inference_with_point
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAM_MODEL = SamModel.from_pretrained("facebook/sam-vit-huge").to(DEVICE)
SAM_PROCESSOR = SamProcessor.from_pretrained("facebook/sam-vit-huge")
FASTSAM_MODEL = FastSAM('FastSAM-x.pt')
EFFICIENT_SAM_MODEL = load(device=DEVICE)

@st.cache_data
def SAM_points_inference(image: np.ndarray, input_points) -> np.ndarray:
    print('Processing SAM... 📊')
    #input_points = [[[float(num) for num in sublist]] for sublist in global_points]
    #print(input_points)
    #input_points = [[[773.0, 167.0]]]
    x = int(input_points[0][0][0])
    y = int(input_points[0][0][1])
    
    inputs = SAM_PROCESSOR(
        Image.fromarray(image),
        input_points=[input_points],
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = SAM_MODEL(**inputs)

    mask = SAM_PROCESSOR.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu()
    )[0][0][0].numpy()
    mask = mask[np.newaxis, ...]
    detections = sv.Detections(xyxy=sv.mask_to_xyxy(masks=mask), mask=mask)
    return detections

@st.cache_data
def FastSAM_points_inference(
    input,
    input_points,
    input_labels,
    input_size=1024, 
    iou_threshold=0.7,
    conf_threshold=0.25
):
    # scaled input points
    #input_points = [[[float(num) for num in sublist]] for sublist in input_points]
    print('Processing FastSAM... 📊')
    results = FASTSAM_MODEL(input,
                    device=DEVICE,
                    retina_masks=True,
                    iou=iou_threshold,
                    conf=conf_threshold,
                    imgsz=input_size)

    prompt_process = FastSAMPrompt(input, results, device=DEVICE)

    # Point prompt
    detections = prompt_process.point_prompt(points=input_points, pointlabel=[1])
    return detections

@st.cache_data
def EfficientSAM_points_inference(image: np.ndarray, input_points):
    x, y = int(input_points[0][0]), int(input_points[0][1])
    point = np.array([[int(x), int(y)]])
    mask = inference_with_point(image, point, EFFICIENT_SAM_MODEL, DEVICE)
    mask = mask[np.newaxis, ...]
    detections = sv.Detections(xyxy=sv.mask_to_xyxy(masks=mask), mask=mask)

    return detections