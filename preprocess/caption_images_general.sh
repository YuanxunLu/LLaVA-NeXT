OUTPUT_ROOT=/mnt/data/dataset_captions
WORK_ROOT=/mnt/data/dataset_captions_work_root

python caption_images_multiprocess.py  \
    --output_root $OUTPUT_ROOT \
    --work_root $WORK_ROOT \
    --bucket s3://objaverse-render-random32view-240516 \
    --dataset_type object \
    --local_prompt_percent 0.15 \
    --download_tar 1 \
    --skip_tar 0 \
    --num_work_tar 16 \
    --load-8bit 0 \
    --load-4bit 0 \
    --process 16 \
    --global_input_image_num 16 \
    --num_global_prompts 1 \