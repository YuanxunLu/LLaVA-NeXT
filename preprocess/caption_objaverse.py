import os
import sys
import glob
import subprocess
import datetime

from multiprocessing import Pool
from time import sleep
import numpy as np
import cv2
import os
import glob
import multiprocessing
import multiprocessing.pool
import subprocess
import time
from typing import Optional
import os
import argparse
from tqdm import tqdm
import warnings
import tarfile
import os
from PIL import Image
import glob
import random
import time
import torch

from llava import conversation as conversation_lib
from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from io import BytesIO
from transformers import TextStreamer


CONDUCTOR_ARGS = '--endpoint-url https://conductor.data.apple.com --cli-read-timeout 300'

def pack_results_single_level(source_folder, target_file, mode='all'):
    if os.path.exists(target_file):
        print(f'file {target_file} exists!')
        return True
    
    # create tar file
    with tarfile.open(target_file, "w") as tar:
        for root, _, files in os.walk(source_folder):
            for file in sorted(files):
                file_path = os.path.join(root, file)
                if file.endswith(('.npy', '.png', 'jpg')):
                    if mode == 'RGB':
                        if not 'depth' in file and not 'normal' in file:
                            arcname = os.path.relpath(file_path, source_folder)
                            tar.add(file_path, arcname=arcname)
                    elif mode == 'depth':
                        if 'depth' in file or '.npy' in file:
                            arcname = os.path.relpath(file_path, source_folder)
                            tar.add(file_path, arcname=arcname)
                    elif mode == 'all':
                        arcname = os.path.relpath(file_path, source_folder)
                        tar.add(file_path, arcname=arcname)
 

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


def load_image(image_file):
    image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
    h, w, c = image.shape
    if c == 4:
        mask = image[..., 3:4] / 255.
        image = image[..., :3] * mask + 255. * (1 - mask)   # replace background as white
    image = image[..., ::-1]  # BGR to RGB
    image = Image.fromarray(image.astype(np.uint8))
    return image


def clear_history(our_chatbot, history):
    our_chatbot.conversation = conv_templates[our_chatbot.conv_mode].copy()

    return None


def bot(our_chatbot, history):
    text=history[-1][0]
    images_this_term=[]
    text_this_term=''
    # import pdb;pdb.set_trace()
    num_new_images = 0
    for i,message in enumerate(history[:-1]):
        if type(message[0]) is tuple:
            images_this_term.append(message[0][0])
            num_new_images+=1
        else:
            num_new_images=0

    assert len(images_this_term)>0, "must have an image"
    # image_files = (args.image_file).split(',')
    # image = [load_image(f) for f in images_this_term if f]
    image_list=[]
    for f in images_this_term:
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



def wrap_process(list_args):
    return process(*list_args)


def queue_worker(
    queue: multiprocessing.JoinableQueue,
    count: multiprocessing.Value,
    gpu: int,
    use_gpu: bool,
    worker_i,
) -> None:
    while True:
        get_data = queue.get()
        if get_data is None:
            break
        global_id, item = get_data
        tar_name = item
        # gpu id
        gpu_id = worker_i % 8
        
        # 1. download tars
        bucket_tar = os.path.join(bucket, tar_name)
        tar_file = os.path.join(work_root, tar_name)
        subprocess.call(['aws', '--endpoint-url', 'https://conductor.data.apple.com', '--cli-read-timeout', '300', 
                         's3', 'cp', bucket_tar, tar_file])
        
        # 2. untar tars
        basename = os.path.splitext(os.path.basename(tar_name))[0]
        tar_root = os.path.join(work_root, basename)
        os.makedirs(tar_root, exist_ok=True)
        with tarfile.open(tar_file, "r:*") as tar:
            tar.extractall(path=tar_root)
            print(f"Tar file {tar_file} extracted to {tar_root}")
            os.remove(tar_file)
        
        # 3. image captions
        output_caption_folder = os.path.join(output_root, basename)
        # load llava at first
        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device_map=gpu_id)
        
        our_chatbot = InferenceDemo(args, args.model_path, tokenizer, model, image_processor, context_len)
    
        all_files = os.listdir(tar_root)
        all_images = [f for f in all_files if f.endswith(('.jpg', '.png')) and 'depth' not in f]
        all_ids = sorted(list(set([file.split('.')[0] for file in all_images])))
    
        DESCRIPTION_PROMPTS = [
            'Please write a detailed image prompt for the object/objects.',
            'Describe the color, texture, shape, and features of the object/objects in the image.',
            'Please describe the object/objects in the image.',
            'Write a image prompt for the object/objects',
            'Write a image prompt for the object/objects, less than 10 words.',
            'Write a image prompt for the object/objects, less than 20 words.',
            'Write a image prompt for the object/objects, less than 50 words.',
            'Write a image prompt for the object/objects, less than 100 words.',
            'Write a image prompt for the object/objects, less than 200 words.'
        ]
    
        for worker_task_id, image_id in enumerate(all_ids):
            image_paths = sorted([os.path.join(tar_root, f) for f in all_images if image_id in f])
        
            ### 1. local caption for each image
            for image_path in image_paths:
                # random select a input prompt
                input_prompt = np.random.choice(DESCRIPTION_PROMPTS)
                # print('Select a prompt:', input_prompt)
                history = [
                    [(image_path,), None],
                    [input_prompt, None]
                ]
                with torch.no_grad():
                    history, outputs = bot(our_chatbot, history)

                # remove quotation mark of captions
                if outputs.startswith('"') and outputs.endswith('"'):
                    outputs = outputs[1:-1]
                if outputs.startswith("'") and outputs.endswith("'"):
                    outputs = outputs[1:-1]
                    
                # print(f'Local caption for {image_path}: {outputs}')
                clear_history(our_chatbot, history)
                # save local captions to txt
                txt_file = os.path.join(output_caption_folder, image_path.split('/')[-1][:-4] + '.txt')
                np.savetxt(txt_file, outputs, fmt='%s')
            
            ### 2. global caption given random images
            if num_global_prompts > 1:
                raise NotImplementedError('save function not implemented. only suuport num_global_prompts == 1 now.')

            global_num = min(global_input_image_num, len(image_paths))
            global_image_paths = np.random.choice(image_paths, replace=False, size=global_num)
            # random select a input prompt
            input_prompt = np.random.choice(DESCRIPTION_PROMPTS)
            # print('Select a prompt:', input_prompt)
            history = [[(global_image_path,), None] for global_image_path in global_image_paths]
            history.append([input_prompt, None])
            with torch.no_grad():
                history, outputs = bot(our_chatbot, history)

            # remove quotation mark of captions
            if outputs.startswith('"') and outputs.endswith('"'):
                outputs = outputs[1:-1]
            if outputs.startswith("'") and outputs.endswith("'"):
                outputs = outputs[1:-1]
                
            # print(f'Global caption for {image_id}: {outputs}')
            clear_history(our_chatbot, history)    
            txt_file = os.path.join(output_caption_folder, image_path.split('/')[-1].split('.')[0] + '.global.txt')
            np.savetxt(txt_file, outputs, fmt='%s')
            print(f'[INFO][worker{worker_i}][{worker_task_id}/{len(all_ids)}][{image_id}] Time takes {time.time() - st} seconds.')                                                         
        
        # 4. tar captions
        caption_tar_file = os.path.join(output_root, tar_name)
        pack_results_single_level(output_caption_folder, caption_tar_file, mode='all')

            

### This script includes 1. download tar files 2. untar the tar file and 3. do image captioning and 4. save results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_root', type=str, default='/media/yuanxun/G/dataset_captions_work_root/')
    parser.add_argument('--output_root', type=str, default='/media/yuanxun/G/dataset_captions/')
    parser.add_argument('--bucket', type=str, default='')
    parser.add_argument('--download_tar', type=int, default=1)
    parser.add_argument('--skip_tar', type=int, default=0)
    parser.add_argument('--num_work_tar', type=int, default=16)
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--process', type=int, default=16)
    
    parser.add_argument("--model_path", default="lmms-lab/llava-next-interleave-qwen-7b", type=str)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", type=int, default=0)
    parser.add_argument("--load-4bit", type=int, default=0)
    parser.add_argument('--global_input_image_num', type=int, default=16)
    parser.add_argument('--num_global_prompts', type=int, default=1)
    args = parser.parse_args()

    
    bucket, download_tar, work_root, output_root, num_process = \
        args.bucket, args.download_tar, args.work_root, args.output_root, args.process
    
    global_input_image_num, num_global_prompts = \
        args.global_input_image_num, args.num_global_prompts
    
    # download tar_list first
    if download_tar:
        bucket_tar_list = os.path.join(bucket, 'tar_list.txt')
        tar_list_file = os.path.join(output_root, 'tar_list.txt')
        subprocess.call(['aws', '--endpoint-url', 'https://conductor.data.apple.com', '--cli-read-timeout', '300', 
                        's3', 'cp', bucket_tar_list, tar_list_file])
        # load tar_list
        with open(tar_list_file, "r") as f:
            tar_list = f.readlines()
        tar_list = [tar.strip() for tar in tar_list]
        st_tar, ed_tar = args.skip_tar, args.skip_tar + args.num_work_tar
        file_list = tar_list[st_tar : ed_tar]
        num_items = len(file_list)
    else:
        raise NotImplementedError()
    
    # multiprocess
    print(f"[INFO] Start processing {num_items} tars")
    print(f"[INFO] Task tar names: {file_list}")

    num_items_CPU = num_items
    print(f'Current Setting: Num GPU process: {num_process}, Num task: {num_items}')
    print(f'============== GPU Tasks: {num_items_CPU} =============')
    
    st = time.time()
    if num_items_CPU > 0:
        queue_CPU = multiprocessing.Queue()
        count_CPU = multiprocessing.Value("i", 0)
        process_list = []
        # Start CPU worker processes
        for worker_i in range(num_process):
            process = multiprocessing.Process(
                target=queue_worker, args=(queue_CPU, count_CPU, 0, False, worker_i)
            )
            process.daemon = True
            process.start()
            process_list.append(process)
            
        for global_id, item in enumerate(file_list):
            queue_CPU.put([global_id, item])
        for _ in range(num_process):
            queue_CPU.put(None)
        for p in process_list:
            p.join()

    ed = time.time()
    print(f'Preprocessing takes {ed - st} seconds')

    


