import os
import subprocess
import numpy as np

bucket = 's3://objaverse-render-random32view-240516'

tar_list_names = ['tar_list_train.txt', 'tar_list_test.txt']
all_tar_files = []
for tar_name in tar_list_names:
    bucket_tar_list = os.path.join(bucket, 'tar_list_train.txt')
    tar_list_file = 'tar_list_train.txt'
    subprocess.call(['aws', '--endpoint-url', 'https://conductor.data.apple.com', '--cli-read-timeout', '300', 
                    's3', 'cp', bucket_tar_list, tar_list_file])
    # load tar_list
    with open(tar_list_file, "r") as f:
        tar_list = f.readlines()
    tar_list = [tar.strip() for tar in tar_list]
    all_tar_files.extend(tar_list)


all_tar_files = list(set(all_tar_files))
np.savetxt('all_tar_files.txt', all_tar_files, fmt='%s')