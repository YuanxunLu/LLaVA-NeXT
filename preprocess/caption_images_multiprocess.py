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
# import torch

from llava import conversation as conversation_lib
from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from io import BytesIO
from transformers import TextStreamer



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
        
        # run the single script file
        command = f'CUDA_VISIBLE_DEVICES={gpu_id} python caption_images_single_general.py --output_root {output_root} --work_root {work_root} --bucket {bucket} --upload_bucket {args.upload_bucket} --tar_file {tar_name} --dataset_type {args.dataset_type} --local_prompt_percent {local_prompt_percent} --load-8bit 0 --load-4bit 0 --global_input_image_num {args.global_input_image_num} --num_global_prompts 1'
        subprocess.run(command, shell=True)
        # print(f'Preprocessing {tar_name} takes {ed - st} seconds')

            

### This script includes 1. download tar files 2. untar the tar file and 3. do image captioning and 4. save results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_root', type=str, default='/media/yuanxun/G/dataset_captions_work_root/')
    parser.add_argument('--output_root', type=str, default='/media/yuanxun/G/dataset_captions/')
    parser.add_argument('--bucket', type=str, default='')
    parser.add_argument('--upload_bucket', type=str, default='')
    parser.add_argument('--download_tar', type=int, default=1)
    parser.add_argument('--skip_tar', type=int, default=0)
    parser.add_argument('--num_work_tar', type=int, default=16)
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--process', type=int, default=16)
    parser.add_argument('--dataset_type', type=str, default='object', help='object or scene')
    parser.add_argument('--local_prompt_percent', type=float, default=0.15)
    
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
    
    global_input_image_num, num_global_prompts, local_prompt_percent = \
        args.global_input_image_num, args.num_global_prompts, args.local_prompt_percent
    
    # download tar_list first
    if download_tar:        
        tar_list_names = ['tar_list_train.txt', 'tar_list_test.txt']
        all_tar_files = []
        for tar_name in tar_list_names:
            bucket_tar_list = os.path.join(bucket, tar_name)
            subprocess.call(['aws', '--endpoint-url', 'https://conductor.data.apple.com', '--cli-read-timeout', '300', 
                            's3', 'cp', bucket_tar_list, tar_name])
            # load tar_list
            with open(tar_name, "r") as f:
                tar_list = f.readlines()
            tar_list = [tar.strip() for tar in tar_list]
            all_tar_files.extend(tar_list)
        tar_list = sorted(list(set(all_tar_files)))
        st_tar, ed_tar = args.skip_tar, args.skip_tar + args.num_work_tar
        file_list = tar_list[st_tar : ed_tar]
        num_items = len(file_list)
    else:
        assert 0 == os.system(f'bash get_unfinished.sh {bucket} {args.upload_bucket}')
        with open('unfinished_tar_files.txt') as f:
            all_tar_files = [l.strip() for l in f if len(l.strip()) > 0]
        tar_list = sorted(list(set(all_tar_files)))
        st_tar, ed_tar = args.skip_tar, args.skip_tar + args.num_work_tar
        file_list = tar_list[st_tar : ed_tar]
        num_items = len(file_list)
    
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

    


