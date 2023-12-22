import streamlit as st
from streamlit_drawable_canvas import st_canvas

from PIL import Image
import pandas as pd
import numpy as np
import torch

from utils.SAM import SAM_points_inference, FastSAM_points_inference, EfficientSAM_points_inference, EdgeSAM_points_inference
from utils.draw import draw_SAM_mask_point, draw_FastSAM_point, draw_EdgeSAM_point
from utils.tools import pil_to_bytes

def click(container_width,height,scale,radius_width,show_mask,im):
    for each in ['color_change_point_box','input_masks_color_box']:
        if each in st.session_state:st.session_state.pop(each)
    canvas_result = st_canvas(
            fill_color="rgba(255, 255, 0, 0.8)",
            background_image = st.session_state['im'],
            drawing_mode='point',
            width = container_width,
            height = height * scale,
            point_display_radius = radius_width,
            stroke_width=2,
            update_streamlit=True,
            key="click",)
    if not show_mask:
        im = Image.fromarray(im).convert("RGB")
        rerun = False
        if im != st.session_state['im']:
            rerun = True
        st.session_state['im'] = im
        if rerun:
            st.rerun()
    elif canvas_result.json_data is not None:
        df = pd.json_normalize(canvas_result.json_data["objects"])
        if len(df) == 0:
            st.session_state.clear()
            if 'canvas_result' not in st.session_state:
                st.session_state['canvas_result'] = len(df)
                st.rerun()
            elif len(df) != st.session_state['canvas_result']:
                st.session_state['canvas_result'] = len(df)
                st.rerun()
            return
        
        df["center_x"] = df["left"]
        df["center_y"] = df["top"]
        
        input_points = []
        input_labels = []
        
        for _, row in df.iterrows():
            x, y = row["center_x"] + 5, row["center_y"]
            x = int(x/scale)
            y = int(y/scale)
            input_points.append([x, y])
            if row['fill'] == "rgba(0, 255, 0, 0.8)":
                input_labels.append(1)
            else:
                input_labels.append(0)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # SAM inference
            SAM_masks = SAM_points_inference(im, [input_points])
            st.image(draw_SAM_mask_point(im, SAM_masks, input_points[0][0], input_points[0][1]))
            st.success('SAM Inference completed!', icon="✅")
            
            # EfficientSAM inference
            EfficientSAM_masks = EfficientSAM_points_inference(im, input_points)
            st.image(draw_SAM_mask_point(im, EfficientSAM_masks, input_points[0][0], input_points[0][1]))
            st.success('EfficientSAM Inference completed!', icon="✅")
        
        with col2:
            # FastSAM inference
            FastSAM_masks = FastSAM_points_inference(im, input_points, input_labels)
            st.image(draw_FastSAM_point(FastSAM_masks))
            st.success('FastSAM Inference completed!', icon="✅")
        
            # EdgeSAM inference
            EdgeSAM_masks = EdgeSAM_points_inference(im, input_points, [1])
            st.image(draw_EdgeSAM_point(im, EdgeSAM_masks))
            st.success('EdgeSAM Inference completed!', icon="✅")
        
        
def main():
    print('init')    
    torch.cuda.empty_cache()
    
    with st.sidebar:
        im = st.file_uploader(label='Upload image',type=['png','jpg','tif'])
        option = st.selectbox(
            'Segmentation mode',
            ('Click', 'Box', 'Everything'))
    
        show_mask = st.checkbox('Show mask',value = True)
        radius_width = st.slider('Radius/Width for Click/Box',0,20,5,1)
        
    if im:
        im = Image.open(im).convert("RGB")
        if 'im' not in st.session_state:
            st.session_state['im'] = im
        width, height   = im.size[:2]
        im              = np.array(im)
        container_width = 700
        scale           = container_width/width
        if option == 'Click':
            click(container_width,
                  height,
                  scale,
                  radius_width,
                  show_mask,
                  im)
    else:
        st.session_state.clear()

if __name__ == '__main__':
    main()