
# from .demo_modelpart import InferenceDemo
# import gradio as gr
import os
# import time
import cv2
import glob
import random
import time

# import copy
import torch
# import random
import numpy as np

from llava import conversation as conversation_lib
from llava.constants import DEFAULT_IMAGE_TOKEN


from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

class InferenceDemo(object):
    def __init__(self,args,model_path,tokenizer, model, image_processor, context_len) -> None:
        disable_torch_init()

        
        self.tokenizer, self.model, self.image_processor, self.context_len = tokenizer, model, image_processor, context_len

        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        elif 'qwen' in model_name.lower():
            conv_mode = "qwen_1_5"
        else:
            conv_mode = "llava_v0"

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print("[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(conv_mode, args.conv_mode, args.conv_mode))
        else:
            args.conv_mode = conv_mode
        self.conv_mode=conv_mode
        self.conversation = conv_templates[args.conv_mode].copy()
        self.num_frames = args.num_frames



def is_valid_video_filename(name):
    video_extensions = ['avi', 'mp4', 'mov', 'mkv', 'flv', 'wmv', 'mjpeg']
    
    ext = name.split('.')[-1].lower()
    
    if ext in video_extensions:
        return True
    else:
        return False

def sample_frames(video_file, num_frames) :
    video = cv2.VideoCapture(video_file)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = total_frames // num_frames
    frames = []
    for i in range(total_frames):
        ret, frame = video.read()
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not ret:
            continue
        if i % interval == 0:
            frames.append(pil_img)
    video.release()
    return frames

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            print('failed to load the image')
    else:
        print('Load image from local file')
        print(image_file)
        # image = Image.open(image_file).convert("RGB")
        image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
        h, w, c = image.shape
        if c == 4:
            mask = image[..., 3:4] / 255.
            image = image[..., :3] * mask + 255. * (1 - mask)   # replace background as white
        image = image[..., ::-1]
        # image = cv2.resize(image, (64, 64), cv2.INTER_CUBIC)
        image = Image.fromarray(image.astype(np.uint8))
        
    return image


def clear_history(history):
    our_chatbot.conversation = conv_templates[our_chatbot.conv_mode].copy()

    return None
def clear_response(history):
    for index_conv in range(1, len(history)):
        # loop until get a text response from our model.
        conv = history[-index_conv]
        if not (conv[0] is None):
            break
    question = history[-index_conv][0]
    history = history[:-index_conv]
    return history, question

# def print_like_dislike(x: gr.LikeData):
#     print(x.index, x.value, x.liked)



def add_message(history, message):
    # history=[]
    global our_chatbot
    if len(history)==0:
        our_chatbot = InferenceDemo(args,model_path,tokenizer, model, image_processor, context_len)
        
    for x in message["files"]:
        history.append(((x,), None))
    if message["text"] is not None:
        history.append((message["text"], None))
    return history, gr.MultimodalTextbox(value=None, interactive=False)

## format of history
## example: 4 images, 1 text
## history = [
#    [(image_path1, ), None], [(image_path2, ), None], [(image_path2, ), None], [(image_path3, ), None], ['text strings', None]
#    ]
## more specifically, each part of history is a tuple of (input(image or text), output(text str)). for the last example, if the model
## outputs a text like "xxxxxxx", and I continue typing, history will be:
## history = [
#    [(image_path1, ), None], [(image_path2, ), None], [(image_path2, ), None], [(image_path3, ), None], ['text strings', 'xxxxxxx'], ['new_text strings', None]
#    ]
# so one could do this again and again;
# note that it may take a lot GPU memory, and may raise OOM errors,
# a real example of history is:
# history [[('/tmp/gradio/58ac8930cf5058ce6182b7e987d0600474de3e47/iphone-15-price-1024x576.jpg',), None], 
# [('/tmp/gradio/039a977e49baaa24222d5d6533ffc4d555a1b5b6/dynamic-island-1024x576.jpg',), None], 
# [('/tmp/gradio/c0685d0f5df1b9ddbd6f6e0d1b7ba2f4cd58b5e0/iphone-15-colors-1024x576.jpg',), None], 
# [('/tmp/gradio/6d30d311aea2b930e68472d17cd5c4c992cfb5da/Iphone-15-Usb-c-charger-1024x576.jpg',), None], 
# [('/tmp/gradio/c32138aed8de4af3a03c4f1e04bfd7baea8fc32b/A-17-processors-1024x576.jpg',), None], 
# ['The images are the PPT of iPhone 15 review. can you summarize the main information?', 'The images are from a presentation slide about the iPhone 15, which is an innovative device with a glass back and aluminum enclosure. It has a dynamic island feature that allows for quick access to flight information and delivery tracking. The phone comes in five amazing colors: Queen Pink, Pearl, Pastel Gray, Light Gray, and Onyx. It also has a USB-C compatible charger that can charge multiple devices at once. The iPhone 15 features powerful processors, including Apple A16 Bionic and Apple A17 Pro.'], 
# ['again?', 'I apologize for the confusion. The images are from a presentation slide about the iPhone 15, which is an innovative device with a glass back and aluminum enclosure. It has a dynamic island feature that allows for quick access to flight information and delivery tracking. The phone comes in five amazing colors: Queen Pink, Pearl, Pastel Gray, Light Gray, and Onyx. It also has a USB-C compatible charger that can charge multiple devices at once. The iPhone 15 features powerful processors, including Apple A16 Bionic and Apple A17 Pro.'], 
# ['more?', None]]

def bot(history):
    text=history[-1][0]
    images_this_term=[]
    text_this_term=''
    # import pdb;pdb.set_trace()
    num_new_images = 0
    for i,message in enumerate(history[:-1]):
        if type(message[0]) is tuple:
            images_this_term.append(message[0][0])
            if is_valid_video_filename(message[0][0]):
                num_new_images+=our_chatbot.num_frames
            else:
                num_new_images+=1
        else:
            num_new_images=0
            
    # for message in history[-i-1:]:
    #     images_this_term.append(message[0][0])

    assert len(images_this_term)>0, "must have an image"
    # image_files = (args.image_file).split(',')
    # image = [load_image(f) for f in images_this_term if f]
    image_list=[]
    for f in images_this_term:
        if is_valid_video_filename(f):
            image_list+=sample_frames(f, our_chatbot.num_frames)
        else:
            image_list.append(load_image(f))
    image_tensor = [our_chatbot.image_processor.preprocess(f, return_tensors="pt")["pixel_values"][0].half().to(our_chatbot.model.device) for f in image_list]

    image_tensor = torch.stack(image_tensor)
    image_token = DEFAULT_IMAGE_TOKEN*num_new_images
    # if our_chatbot.model.config.mm_use_im_start_end:
    #     inp = DEFAULT_IM_START_TOKEN + image_token + DEFAULT_IM_END_TOKEN + "\n" + inp
    # else:
    inp=text
    inp = image_token+ "\n" + inp
    our_chatbot.conversation.append_message(our_chatbot.conversation.roles[0], inp)
    # image = None
    our_chatbot.conversation.append_message(our_chatbot.conversation.roles[1], None)
    prompt = our_chatbot.conversation.get_prompt()

    input_ids = tokenizer_image_token(prompt, our_chatbot.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(our_chatbot.model.device)
    stop_str = our_chatbot.conversation.sep if our_chatbot.conversation.sep_style != SeparatorStyle.TWO else our_chatbot.conversation.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, our_chatbot.tokenizer, input_ids)
    streamer = TextStreamer(our_chatbot.tokenizer, skip_prompt=True, skip_special_tokens=True)
    # import pdb;pdb.set_trace()
    with torch.inference_mode():
        output_ids = our_chatbot.model.generate(input_ids, images=image_tensor, do_sample=True, temperature=0.2, max_new_tokens=1024, streamer=streamer, use_cache=False, stopping_criteria=[stopping_criteria])

    outputs = our_chatbot.tokenizer.decode(output_ids[0]).strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    our_chatbot.conversation.messages[-1][-1] = outputs
   
    history[-1]=[text,outputs]
    
    return history, outputs


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--server_name", default="0.0.0.0", type=str)
    argparser.add_argument("--port", default="6123", type=str)
    argparser.add_argument("--model_path", default="lmms-lab/llava-next-interleave-qwen-7b", type=str)
    argparser.add_argument("--model-base", type=str, default=None)
    argparser.add_argument("--num-gpus", type=int, default=1)
    argparser.add_argument("--conv-mode", type=str, default=None)
    argparser.add_argument("--temperature", type=float, default=0.2)
    argparser.add_argument("--max-new-tokens", type=int, default=512)
    argparser.add_argument("--num_frames", type=int, default=16)
    # argparser.add_argument("--load-8bit", action="store_true")
    # argparser.add_argument("--load-4bit", action="store_true")
    argparser.add_argument("--load-8bit", type=int, default=0)
    argparser.add_argument("--load-4bit", type=int, default=1)
    argparser.add_argument("--debug", action="store_true")
    
    args = argparser.parse_args()
    model_path = args.model_path
    filt_invalid="cut"
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit)
    
    our_chatbot = InferenceDemo(args,model_path,tokenizer, model, image_processor, context_len)
    
    ## the data folder structure format of the image files
    ## flat images in a folders
    
    ## first genrate local captions for each images, then gloal captions given random images
    
    unzipped_tar_root = 'examples/realestate10k'
    all_files = os.listdir(unzipped_tar_root)
    all_images = [f for f in all_files if f.endswith(('.jpg', '.png')) and 'depth' not in f]
    all_ids = sorted(list(set([file.split('.')[0] for file in all_images])))
    
    DESCRIPTION_PROMPTS = [
        'Please write a detailed image prompt for the entire scene.',
        'Describe the colors, textures, shapes, and features of everything in the image, including the background and surroundings.',
        'Please describe all the elements in the image, focusing on both objects and the overall scene.',
        'Write a detailed image prompt for the scene.',
        'Write an image prompt for the scene, less than 10 words.',
        'Write an image prompt for the scene, less than 20 words.',
        'Write an image prompt for the scene, less than 50 words.',
        'Write an image prompt for the scene, less than 100 words.',
        'Describe the setting, lighting, and mood of the scene along with the objects within it.'
    ]

    all_ids = all_ids[:1]
    global_input_image_num = 8
    num_global_prompts = 1
    for image_id in all_ids:
        st = time.time()
        image_paths = sorted([os.path.join(unzipped_tar_root, f) for f in all_images if image_id in f])
    
        ### 1. local caption for each image
        for image_path in image_paths:
            # random select a input prompt
            input_prompt = np.random.choice(DESCRIPTION_PROMPTS)
            print('Select a prompt:', input_prompt)
            history = [
                [(image_path,), None],
                [input_prompt, None]
            ]
            with torch.no_grad():
                history, outputs = bot(history)

            # remove quotation mark of captions
            if outputs.startswith('"') and outputs.endswith('"'):
                outputs = outputs[1:-1]
            if outputs.startswith("'") and outputs.endswith("'"):
                outputs = outputs[1:-1]
                
            print(f'Local caption for {image_path}: {outputs}')
            clear_history(history)
        
        ### 2. global caption given random images
        for i in range(num_global_prompts):
            global_image_paths = np.random.choice(image_paths, replace=False, size=global_input_image_num)
            # random select a input prompt
            input_prompt = np.random.choice(DESCRIPTION_PROMPTS)
            print('Select a prompt:', input_prompt)
            history = [[(global_image_path,), None] for global_image_path in global_image_paths]
            history.append([input_prompt, None])
            with torch.no_grad():
                history, outputs = bot(history)

            # remove quotation mark of captions
            if outputs.startswith('"') and outputs.endswith('"'):
                outputs = outputs[1:-1]
            if outputs.startswith("'") and outputs.endswith("'"):
                outputs = outputs[1:-1]
                
            print(f'Global caption for {image_id}: {outputs}')
            clear_history(history)     
        print(f'Time takes for {image_id}: {time.time() - st} seconds.')                                                         
                
            
            
