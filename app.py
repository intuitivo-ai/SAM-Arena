# Thanks to the following repos:
# https://huggingface.co/spaces/An-619/FastSAM/blob/main/app_gradio.py
# https://huggingface.co/spaces/SkalskiP/EfficientSAM
from typing import Tuple

from ultralytics import YOLO
from PIL import ImageDraw
from PIL import Image
import gradio as gr
import numpy as np
import torch

from transformers import SamModel, SamProcessor

import supervision as sv
from utils.tools_gradio import fast_process
from utils.tools import format_results, point_prompt
from utils.draw import draw_circle, calculate_dynamic_circle_radius
from utils.efficient_sam import load, inference_with_box, inference_with_point

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the pre-trained models
FASTSAM_MODEL = YOLO('FastSAM-s.pt')
SAM_MODEL = SamModel.from_pretrained("facebook/sam-vit-huge").to(DEVICE)
SAM_PROCESSOR = SamProcessor.from_pretrained("facebook/sam-vit-huge")
EFFICIENT_SAM_MODEL = load(device=DEVICE)

MASK_COLOR = sv.Color.from_hex("#FF0000")
PROMPT_COLOR = sv.Color.from_hex("#D3D3D3")
MASK_ANNOTATOR = sv.MaskAnnotator(
    color=MASK_COLOR,
    color_lookup=sv.ColorLookup.INDEX)

title = "<center><strong><font size='8'>🤗 Segment Anything Model Arena ⚔️</font></strong></center>"

description = "<center><font size='4'>This is a demo of the <strong>Segment Anything Model Arena</strong>, a collection of models for segmenting anything. "

css = "h1 { text-align: center } .about { text-align: justify; padding-left: 10%; padding-right: 10%; }"

#examples = [["examples/retail01.png"], ["examples/vend01.png"], ["examples/vend02.png"]]

POINT_EXAMPLES = [
    ['https://media.roboflow.com/efficient-sam/corgi.jpg', 1291, 751],
    ['https://media.roboflow.com/efficient-sam/horses.jpg', 1168, 939],
    ['https://media.roboflow.com/efficient-sam/bears.jpg', 913, 1051]
]

#default_example = examples[0]

def annotate_image_with_point_prompt_result(
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

def SAM_points_inference(image: np.ndarray) -> np.ndarray:
    global global_points
    input_points = [[[float(num) for num in sublist]] for sublist in global_points]
    print(input_points)
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

    return annotate_image_with_point_prompt_result(
        image=image, detections=detections, x=x, y=y)
 
def FastSAM_points_inference(
    input,
    input_size=1024, 
    iou_threshold=0.7,
    conf_threshold=0.25,
    better_quality=False,
    withContours=True,
    use_retina=True,
    mask_random_color=True,
):
    global global_points
    global global_point_label
    input = Image.fromarray(input)
    input_size = int(input_size)  # 确保 imgsz 是整数
    # Thanks for the suggestion by hysts in HuggingFace.
    w, h = input.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    input = input.resize((new_w, new_h))
    
    scaled_points = [[int(x * scale) for x in point] for point in global_points]

    results = FASTSAM_MODEL(input,
                    device=DEVICE,
                    retina_masks=True,
                    iou=iou_threshold,
                    conf=conf_threshold,
                    imgsz=input_size,)
    
    results = format_results(results[0], 0)
    annotations, _ = point_prompt(results, scaled_points, global_point_label, new_h, new_w)
    annotations = np.array([annotations])

    fig = fast_process(annotations=annotations,
                       image=input,
                       device=DEVICE,
                       scale=(1024 // input_size),
                       better_quality=better_quality,
                       mask_random_color=mask_random_color,
                       bbox=None,
                       use_retina=use_retina,
                       withContours=withContours,)
    
    global_points = []
    global_point_label = []
    
    return fig

def EfficientSAM_points_inference(image: np.ndarray):
    x, y = int(global_points[0][0]), int(global_points[0][1])
    point = np.array([[int(x), int(y)]])
    mask = inference_with_point(image, point, EFFICIENT_SAM_MODEL, DEVICE)
    mask = mask[np.newaxis, ...]
    detections = sv.Detections(xyxy=sv.mask_to_xyxy(masks=mask), mask=mask)

    return annotate_image_with_point_prompt_result(image=image, detections=detections, x=x, y=y)
    
def get_points_with_draw(image, label, evt: gr.SelectData):
    global global_points
    global global_point_label

    x, y = evt.index[0], evt.index[1]
    point_radius, point_color = 15, (255, 0, 0) if label == 'Add Mask' else (255, 0, 255)
    global_points.append([x, y])
    global_point_label.append(1 if label == 'Add Mask' else 0)
    
    print(x, y, label == 'Add Mask')
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    draw.ellipse([(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)], fill=point_color)
    return image

def clear(_: np.ndarray) -> Tuple[None, None, None, None]:
    return None, None, None, None

gr_input_image = gr.Image(label="Input", value='examples/fruits.jpg')

fast_sam_segmented_image = gr.Image(label="Fast SAM", interactive=False, type='pil')

edge_sam_segmented_imaged = gr.Image(label="Edge SAM", interactive=False, type='pil')


global_points = []
global_point_label = []

with gr.Blocks() as demo:
    with gr.Tab("Points prompt"):
        # Input Image
        with gr.Row(variant="panel"):
            with gr.Column(scale=1, min_width="320", variant="compact"):
                gr_input_image.render()
        
        # Submit & Clear
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    add_or_remove = gr.Radio(["Add Mask", "Remove Area"], value="Add Mask", label="Point label (foreground/background)")
                    with gr.Column():
                        inference_point_button = gr.Button("Segment", variant='primary')
                        clear_button = gr.Button("Clear points", variant='secondary')
        
        # Segment Results Grid  
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                sam_segmented_image = gr.Image(label="SAM")
            with gr.Column(scale=1):
                efficient_sam_segmented_image = gr.Image(label="Efficient SAM")
        
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                fast_sam_segmented_image.render()
            with gr.Column(scale=1):
                edge_sam_segmented_imaged.render()
    
    gr.Markdown("AI Generated Examples")
    # gr.Examples(examples=examples,
    #             inputs=[gr_input_image],
    #             # outputs=sam_segmented_image,
    #             # fn=segment_with_points,
    #             # cache_examples=True,
    #             examples_per_page=3)
    
    gr_input_image.select(get_points_with_draw, [gr_input_image, add_or_remove], gr_input_image)

    inference_point_button.click(
        SAM_points_inference,
        inputs=[gr_input_image],
        outputs=[sam_segmented_image]
    )
    
    inference_point_button.click(
        EfficientSAM_points_inference,
        inputs=[gr_input_image],
        outputs=[efficient_sam_segmented_image])
    
    inference_point_button.click(
        FastSAM_points_inference,
        inputs=[gr_input_image],
        outputs=[fast_sam_segmented_image])
    
    # inference_point_button.click(
    #     EdgeSAM_points_inference,
    #     inputs=[gr_input_image],
    #     outputs=[fast_sam_segmented_image, gr_input_image])
    
    gr_input_image.change(
        clear,
        inputs=gr_input_image,
        outputs=[efficient_sam_segmented_image, sam_segmented_image, fast_sam_segmented_image]
    )

    clear_button.click(clear, outputs=[gr_input_image, efficient_sam_segmented_image, sam_segmented_image, fast_sam_segmented_image])


demo.queue()
demo.launch(debug=True, show_error=True)