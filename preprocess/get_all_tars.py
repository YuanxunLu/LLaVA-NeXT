import os
import subprocess
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('bucket', type=str, default='s3://objaverse-render-random32view-240516')
parser.add_argument('--out', type=str, default='all_tar_files.txt')
args = parser.parse_args()

bucket = args.bucket

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


all_tar_files = list(set(all_tar_files))
np.savetxt(args.out, all_tar_files, fmt='%s')