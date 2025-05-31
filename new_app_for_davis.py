import gradio as gr
import argparse
import gdown
import cv2
import numpy as np
import os
import sys
sys.path.append(sys.path[0]+"/tracker")
sys.path.append(sys.path[0]+"/tracker/model")
from new_track_anything import NewTrackAnything
from new_track_anything import parse_augment
import requests
import json
import torchvision
import torch 
from tools.painter import mask_painter
import psutil
import time

from PIL import Image
import numpy as np

import glob
from pathlib import Path

# download checkpoints
def download_checkpoint(url, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    if not os.path.exists(filepath):
        print("download checkpoints ......")
        response = requests.get(url, stream=True)
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print("download successfully!")

    return filepath

def download_checkpoint_from_google_drive(file_id, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    if not os.path.exists(filepath):
        print("Downloading checkpoints from Google Drive... tips: If you cannot see the progress bar, please try to download it manuall \
              and put it in the checkpointes directory. E2FGVI-HQ-CVPR22.pth: https://github.com/MCG-NKU/E2FGVI(E2FGVI-HQ model)")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filepath, quiet=False)
        print("Downloaded successfully!")

    return filepath

# convert points input to prompt state
def get_prompt(click_state, click_input):
    inputs = json.loads(click_input)
    points = click_state[0]
    labels = click_state[1]
    for input in inputs:
        points.append(input[:2])
        labels.append(input[2])
    click_state[0] = points
    click_state[1] = labels
    prompt = {
        "prompt_type":["click"],
        "input_point":click_state[0],
        "input_label":click_state[1],
        "multimask_output":"True",
    }
    return prompt

def load_frames_from_dir(dir_path: str):
    """
    dir_path 안의 이미지 파일을 읽어 RGB numpy array 리스트로 반환.
    파일명이 00001.jpg 처럼 정렬돼 있으면 자연스럽게 정렬됩니다.
    """
    # 확장자 원하는 대로 추가
    img_files = sorted(
        glob.glob(os.path.join(dir_path, "*.[jp][pn]g")) +   # jpg, jpeg, png
        glob.glob(os.path.join(dir_path, "*.bmp"))
    )

    if len(img_files) == 0:
        raise ValueError(f"No images found in {dir_path}")

    frames = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in img_files]
    return frames

def get_frames_from_source(source_path, video_state):
    """
    - source_path가 폴더면 이미지 시퀀스
    - 파일이면 동영상
    """
    frames = []
    user_name = time.time()
    operation_log = [("",""),("Source loaded. Click the image to add targets.","Normal")]

    print("===================================")
    print("Loading frames from source:", source_path)
    print("===================================")

    # ① 폴더인지 먼저 체크
    if os.path.isdir(source_path):
        frames = load_frames_from_dir(source_path)
        fps = 30  # 이미지 시퀀스는 FPS 정보가 없으므로 기본값, slider로 바꾸려면 여기서 노출
    else:
        # ② 기존 비디오 로드 루트
        cap = cv2.VideoCapture(source_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            current_memory_usage = psutil.virtual_memory().percent
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if current_memory_usage > 90:
                operation_log = [("Memory >90 %, stop loading. Reduce resolution/FPS.", "Error")]
                break
        cap.release()

    if len(frames) == 0:
        raise ValueError("No frames were loaded – check the path or file type.")

    image_size = frames[0].shape[:2]
    video_state = {
        "user_name": user_name,
        "video_name": os.path.basename(source_path),
        "origin_images": frames,
        "painted_images": frames.copy(),
        "masks": [np.zeros(image_size, np.uint8)] * len(frames),
        "logits": [None] * len(frames),
        "select_frame_number": 0,
        "fps": fps,
    }
    video_info = f"Source: {video_state['video_name']} | FPS: {fps} | Frames: {len(frames)} | Size: {image_size}"
    # SAM 초기화
    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(frames[0])

    return (
        video_state, video_info, frames[0],
        gr.update(visible=True, maximum=len(frames), value=1),
        gr.update(visible=True, maximum=len(frames), value=len(frames)),
        gr.update(visible=True), gr.update(visible=True), gr.update(visible=True),
        gr.update(visible=True), gr.update(visible=True), gr.update(visible=True),
        gr.update(visible=True), gr.update(visible=True), gr.update(visible=True),
        gr.update(visible=True, value=operation_log)
    )


# get the select frame from gradio slider
def select_template(image_selection_slider, video_state, interactive_state, mask_dropdown):

    # images = video_state[1]
    image_selection_slider -= 1
    video_state["select_frame_number"] = image_selection_slider

    # once select a new template frame, set the image in sam

    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][image_selection_slider])

    # update the masks when select a new template frame
    # if video_state["masks"][image_selection_slider] is not None:
        # video_state["painted_images"][image_selection_slider] = mask_painter(video_state["origin_images"][image_selection_slider], video_state["masks"][image_selection_slider])
    if mask_dropdown:
        print("ok")
    operation_log = [("",""), ("Select frame {}. Try click image and add mask for tracking.".format(image_selection_slider),"Normal")]


    return video_state["painted_images"][image_selection_slider], video_state, interactive_state, operation_log

# set the tracking end frame
def get_end_number(track_pause_number_slider, video_state, interactive_state):
    interactive_state["track_end_number"] = track_pause_number_slider
    operation_log = [("",""),("Set the tracking finish at frame {}".format(track_pause_number_slider),"Normal")]

    return video_state["painted_images"][track_pause_number_slider],interactive_state, operation_log

# use sam to get the mask
def sam_refine(video_state, point_prompt, click_state, interactive_state, evt:gr.SelectData):
    """
    Args:
        template_frame: PIL.Image
        point_prompt: flag for positive or negative button click
        click_state: [[points], [labels]]
    """
    if point_prompt == "Positive":
        coordinate = "[[{},{},1]]".format(evt.index[0], evt.index[1])
        interactive_state["positive_click_times"] += 1
    else:
        coordinate = "[[{},{},0]]".format(evt.index[0], evt.index[1])
        interactive_state["negative_click_times"] += 1
    
    # prompt for sam model
    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][video_state["select_frame_number"]])
    prompt = get_prompt(click_state=click_state, click_input=coordinate)

    mask, logit, painted_image = model.first_frame_click( 
                                                      image=video_state["origin_images"][video_state["select_frame_number"]], 
                                                      points=np.array(prompt["input_point"]),
                                                      labels=np.array(prompt["input_label"]),
                                                      multimask=prompt["multimask_output"],
                                                      )
    video_state["masks"][video_state["select_frame_number"]] = mask
    video_state["logits"][video_state["select_frame_number"]] = logit
    video_state["painted_images"][video_state["select_frame_number"]] = painted_image

    operation_log = [("",""), ("Use SAM for segment. You can try add positive and negative points by clicking. Or press Clear clicks button to refresh the image. Press Add mask button when you are satisfied with the segment","Normal")]
    return painted_image, video_state, interactive_state, operation_log

def add_multi_mask(video_state, interactive_state, mask_dropdown):
    try:
        mask = video_state["masks"][video_state["select_frame_number"]]
        interactive_state["multi_mask"]["masks"].append(mask)
        interactive_state["multi_mask"]["mask_names"].append("mask_{:03d}".format(len(interactive_state["multi_mask"]["masks"])))
        mask_dropdown.append("mask_{:03d}".format(len(interactive_state["multi_mask"]["masks"])))
        select_frame, run_status = show_mask(video_state, interactive_state, mask_dropdown)

        operation_log = [("",""),("Added a mask, use the mask select for target tracking.","Normal")]
    except:
        operation_log = [("Please click the left image to generate mask.", "Error"), ("","")]
    return interactive_state, gr.update(choices=interactive_state["multi_mask"]["mask_names"], value=mask_dropdown), select_frame, [[],[]], operation_log

def clear_click(video_state, click_state):
    click_state = [[],[]]
    template_frame = video_state["origin_images"][video_state["select_frame_number"]]
    operation_log = [("",""), ("Clear points history and refresh the image.","Normal")]
    return template_frame, click_state, operation_log

def remove_multi_mask(interactive_state, mask_dropdown):
    interactive_state["multi_mask"]["mask_names"]= []
    interactive_state["multi_mask"]["masks"] = []

    operation_log = [("",""), ("Remove all mask, please add new masks","Normal")]
    return interactive_state, gr.update(choices=[],value=[]), operation_log

def show_mask(video_state, interactive_state, mask_dropdown):
    mask_dropdown.sort()
    select_frame = video_state["origin_images"][video_state["select_frame_number"]]
    for i in range(len(mask_dropdown)):
        mask_number = int(mask_dropdown[i].split("_")[1]) - 1
        mask = interactive_state["multi_mask"]["masks"][mask_number]
        select_frame = mask_painter(select_frame, mask.astype('uint8'), mask_color=mask_number+2)
    
    operation_log = [("",""), ("Select {} for tracking".format(mask_dropdown),"Normal")]
    return select_frame, operation_log

# tracking vos
def vos_tracking_video(video_state, interactive_state, mask_dropdown):
    operation_log = [("",""), ("Track the selected masks.","Normal")]
    model.xmem.clear_memory()
    if interactive_state["track_end_number"]:
        following_frames = video_state["origin_images"][video_state["select_frame_number"]:interactive_state["track_end_number"]]
    else:
        following_frames = video_state["origin_images"][video_state["select_frame_number"]:]

    if interactive_state["multi_mask"]["masks"]:
        if len(mask_dropdown) == 0:
            mask_dropdown = ["mask_001"]
        mask_dropdown.sort()
        template_mask = interactive_state["multi_mask"]["masks"][int(mask_dropdown[0].split("_")[1]) - 1] * (int(mask_dropdown[0].split("_")[1]))
        for i in range(1,len(mask_dropdown)):
            mask_number = int(mask_dropdown[i].split("_")[1]) - 1 
            template_mask = np.clip(template_mask+interactive_state["multi_mask"]["masks"][mask_number]*(mask_number+1), 0, mask_number+1)
        video_state["masks"][video_state["select_frame_number"]]= template_mask
    else:      
        template_mask = video_state["masks"][video_state["select_frame_number"]]
    fps = video_state["fps"]

    # operation error
    if len(np.unique(template_mask))==1:
        template_mask[0][0]=1
        operation_log = [("Error! Please add at least one mask to track by clicking the left image.","Error"), ("","")]
        # return video_output, video_state, interactive_state, operation_error
    masks, logits, painted_images = model.generator(images=following_frames, template_mask=template_mask)
    # clear GPU memory
    model.xmem.clear_memory()

    if interactive_state["track_end_number"]: 
        video_state["masks"][video_state["select_frame_number"]:interactive_state["track_end_number"]] = masks
        video_state["logits"][video_state["select_frame_number"]:interactive_state["track_end_number"]] = logits
        video_state["painted_images"][video_state["select_frame_number"]:interactive_state["track_end_number"]] = painted_images
    else:
        video_state["masks"][video_state["select_frame_number"]:] = masks
        video_state["logits"][video_state["select_frame_number"]:] = logits
        video_state["painted_images"][video_state["select_frame_number"]:] = painted_images

    out_name   = Path(video_state["video_name"]).stem   
    out_path   = f"./result_new/{out_name}.mp4"
    video_output = generate_video_from_frames(
        video_state["painted_images"],
        output_path=out_path,
        fps=fps,
    )

    interactive_state["inference_times"] += 1
    
    print("For generating this tracking result, inference times: {}, click times: {}, positive: {}, negative: {}".format(interactive_state["inference_times"], 
                                                                                                                                           interactive_state["positive_click_times"]+interactive_state["negative_click_times"],
                                                                                                                                           interactive_state["positive_click_times"],
                                                                                                                                        interactive_state["negative_click_times"]))

    if interactive_state["mask_save"]:
        output_mask_path = './result_new/{}'.format(video_state["video_name"].split('.')[0])
        if not os.path.exists(output_mask_path):
            os.makedirs(output_mask_path)
        print("====================================")
        print("Save masks to {}".format(output_mask_path))
        print("====================================")
        for i, m in enumerate(video_state['masks']):
            img = Image.fromarray(m * 255)
            img.save(os.path.join(output_mask_path, f'{i:05d}.png'), format='PNG')


    return video_output, video_state, interactive_state, operation_log


# generate video after vos inference
def generate_video_from_frames(frames, output_path, fps=30):
    """
    Generates a video from a list of frames.
    
    Args:
        frames (list of numpy arrays): The frames to include in the video.
        output_path (str): The path to save the generated video.
        fps (int, optional): The frame rate of the output video. Defaults to 30.
    """
    video = torch.from_numpy(np.asarray(frames)).to(torch.uint8)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchvision.io.write_video(
        output_path,
        video,
        fps=fps,                 # ← **확실히 파이썬 int**
        video_codec="libx264"
    )
    return output_path


# args, defined in track_anything.py
args = parse_augment()

# check and download checkpoints if needed
SAM_checkpoint_dict = {
    'vit_h': "sam_vit_h_4b8939.pth",
    'vit_l': "sam_vit_l_0b3195.pth", 
    "vit_b": "sam_vit_b_01ec64.pth"
}
SAM_checkpoint_url_dict = {
    'vit_h': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    'vit_l': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    'vit_b': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
}
sam_checkpoint = SAM_checkpoint_dict[args.sam_model_type] 
sam_checkpoint_url = SAM_checkpoint_url_dict[args.sam_model_type] 
xmem_checkpoint = "XMem-s012.pth"
xmem_checkpoint_url = "https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth"
e2fgvi_checkpoint = "E2FGVI-HQ-CVPR22.pth"
e2fgvi_checkpoint_id = "10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3"


folder ="./checkpoints"
SAM_checkpoint = download_checkpoint(sam_checkpoint_url, folder, sam_checkpoint)
xmem_checkpoint = download_checkpoint(xmem_checkpoint_url, folder, xmem_checkpoint)
e2fgvi_checkpoint = download_checkpoint_from_google_drive(e2fgvi_checkpoint_id, folder, e2fgvi_checkpoint)
args.port = 12212
args.device = "cuda:0"
args.mask_save = True

# initialize sam, xmem, e2fgvi models
model = NewTrackAnything(SAM_checkpoint, xmem_checkpoint, e2fgvi_checkpoint,args)


title = """<p><h1 align="center">Track-Anything</h1></p>
    """
description = """<p>Gradio demo for Track Anything, a flexible and interactive tool for video object tracking and segmentation. I To use it, simply upload your video, or click one of the examples to load them. Code: <a href="https://github.com/gaomingqi/Track-Anything">https://github.com/gaomingqi/Track-Anything</a> <a href="https://huggingface.co/spaces/watchtowerss/Track-Anything?duplicate=true"><img style="display: inline; margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space" /></a></p>"""


with gr.Blocks() as iface:
    """
        state for 
    """
    click_state = gr.State([[],[]])
    interactive_state = gr.State({
        "inference_times": 0,
        "negative_click_times" : 0,
        "positive_click_times": 0,
        "mask_save": args.mask_save,
        "multi_mask": {
            "mask_names": [],
            "masks": []
        },
        "track_end_number": None,
        "resize_ratio": 1
    }
    )

    video_state = gr.State(
        {
        "user_name": "",
        "video_name": "",
        "origin_images": None,
        "painted_images": None,
        "masks": None,
        "logits": None,
        "select_frame_number": 0,
        "fps": 30
        }
    )
    gr.Markdown(title)
    gr.Markdown(description)
    with gr.Row():

        # for user video input
        with gr.Column():
            with gr.Row(scale=0.4):

                source_input = gr.Textbox(label="Video file OR Folder with images")

                with gr.Column():
                    video_info = gr.Textbox(label="Video Info")
          
            with gr.Row():
                # put the template frame under the radio button
                with gr.Column():
                    # extract frames
                    with gr.Column():
                        extract_frames_button = gr.Button(value="Get video info", interactive=True, variant="primary") 

                     # click points settins, negative or positive, mode continuous or single
                    with gr.Row():
                        with gr.Row():
                            point_prompt = gr.Radio(
                                choices=["Positive",  "Negative"],
                                value="Positive",
                                label="Point prompt",
                                interactive=True,
                                visible=False)
                            remove_mask_button = gr.Button(value="Remove mask", interactive=True, visible=False) 
                            clear_button_click = gr.Button(value="Clear clicks", interactive=True, visible=False)
                            # clear_button_click = gr.Button(value="Clear clicks", interactive=True, visible=False).style(height=160)
                            Add_mask_button = gr.Button(value="Add mask", interactive=True, visible=False)
                    template_frame = gr.Image(type="pil",interactive=True, elem_id="template_frame", visible=False)
                    # template_frame = gr.Image(type="pil",interactive=True, elem_id="template_frame", visible=False).style(height=360)
                    image_selection_slider = gr.Slider(minimum=1, maximum=100, step=1, value=1, label="Track start frame", visible=False)
                    track_pause_number_slider = gr.Slider(minimum=1, maximum=100, step=1, value=1, label="Track end frame", visible=False)
            
                with gr.Column():
                    run_status = gr.HighlightedText(value=[("Text","Error"),("to be","Label 2"),("highlighted","Label 3")], visible=False)
                    mask_dropdown = gr.Dropdown(multiselect=True, value=[], label="Mask selection", info=".", visible=False)
                    video_output = gr.Video(visible=False)
                    # video_output = gr.Video(autosize=True, visible=False).style(height=360)
                    with gr.Row():
                        tracking_video_predict_button = gr.Button(value="Tracking", visible=False)

    # first step: get the video information 
    extract_frames_button.click(
        fn=get_frames_from_source,
        inputs=[source_input, video_state],
        outputs=[video_state, video_info, template_frame,
                 image_selection_slider, track_pause_number_slider,point_prompt, clear_button_click, Add_mask_button, template_frame,
                 tracking_video_predict_button, video_output, mask_dropdown, remove_mask_button, run_status]
    )   

    # second step: select images from slider
    image_selection_slider.release(fn=select_template, 
                                   inputs=[image_selection_slider, video_state, interactive_state], 
                                   outputs=[template_frame, video_state, interactive_state, run_status], api_name="select_image")
    track_pause_number_slider.release(fn=get_end_number, 
                                   inputs=[track_pause_number_slider, video_state, interactive_state], 
                                   outputs=[template_frame, interactive_state, run_status], api_name="end_image")
    
    # click select image to get mask using sam
    template_frame.select(
        fn=sam_refine,
        inputs=[video_state, point_prompt, click_state, interactive_state],
        outputs=[template_frame, video_state, interactive_state, run_status]
    )

    # add different mask
    Add_mask_button.click(
        fn=add_multi_mask,
        inputs=[video_state, interactive_state, mask_dropdown],
        outputs=[interactive_state, mask_dropdown, template_frame, click_state, run_status]
    )

    remove_mask_button.click(
        fn=remove_multi_mask,
        inputs=[interactive_state, mask_dropdown],
        outputs=[interactive_state, mask_dropdown, run_status]
    )

    # tracking video from select image and mask
    tracking_video_predict_button.click(
        fn=vos_tracking_video,
        inputs=[video_state, interactive_state, mask_dropdown],
        outputs=[video_output, video_state, interactive_state, run_status]
    )

    # click to get mask
    mask_dropdown.change(
        fn=show_mask,
        inputs=[video_state, interactive_state, mask_dropdown],
        outputs=[template_frame, run_status]
    )
    
    # clear input

    def _reset_states():
        """영상·폴더 경로를 지우면 전체 UI와 state 초기화"""
        return (
            {   # video_state
                "user_name": "",
                "video_name": "",
                "origin_images": None,
                "painted_images": None,
                "masks": None,
                "logits": None,
                "select_frame_number": 0,
                "fps": 30,
            },
            {   # interactive_state
                "inference_times": 0,
                "negative_click_times": 0,
                "positive_click_times": 0,
                "mask_save": args.mask_save,
                "multi_mask": {"mask_names": [], "masks": []},
                "track_end_number": 0,
                "resize_ratio": 1,
            },
            [[], []],          # click_state
            None,              # video_output
            None,              # template_frame
            *[gr.update(visible=False) for _ in range(9)],           # 버튼·슬라이더 9개
            gr.update(visible=False, value=[]),  # mask_dropdown
            gr.update(visible=False),            # remove_mask_button
            gr.update(visible=False),            # run_status
        )

    source_input.change(
        fn=_reset_states,
        inputs=None,          # 입력 값은 안 써도 되므로 None
        outputs=[
            video_state, interactive_state, click_state,
            video_output, template_frame,
            tracking_video_predict_button, image_selection_slider,
            track_pause_number_slider, point_prompt, clear_button_click,
            Add_mask_button, template_frame, tracking_video_predict_button,
            video_output, mask_dropdown, remove_mask_button, run_status,
        ],
        queue=False,
        show_progress=False,
    )

    # points clear
    clear_button_click.click(
        fn = clear_click,
        inputs = [video_state, click_state,],
        outputs = [template_frame,click_state, run_status],
    )

iface.queue(default_concurrency_limit=1)
iface.launch(share=True, debug=True, server_port=args.port, server_name="0.0.0.0")