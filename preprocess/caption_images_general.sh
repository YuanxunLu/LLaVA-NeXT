OUTPUT_ROOT=/media/yuanxun/G/dataset_captions
WORK_ROOT=/media/yuanxun/G/dataset_captions_work_root

python caption_images_general.py  \
    --output_root $OUTPUT_ROOT \
    --work_root $WORK_ROOT \
    --bucket objaversexxxx \
    --download_tar 1 \
    --skip_tar 0 \
    --num_work_tar 16 \
    --load-8bit 0 \
    --load-4bit 0 \
    --process 16 \
    --global_input_image_num 16 \
    --num_global_prompts 1 \