cd ..
python test_llava_interleave_mvcaption.py
cd -

python caption_images_multiprocess.py  \
    --output_root ${PROC_OUTPUT_ROOT:-"/mnt/data/dataset_captions"} \
    --work_root ${PROC_WORK_ROOT:-"/mnt/data/dataset_captions_work_root"} \
    --bucket ${PROC_BUCKET:-"s3://objaverse-render-random32view-240516"} \
    --upload_bucket ${PROC_UPLOAD:-"s3://objaverse-caption-random32view-240516"} \
    --dataset_type ${PROC_TYPE:-"object"} \
    --local_prompt_percent ${PROC_PERCENT:-"0.15"} \
    --download_tar ${PROC_TARLIST:-1} \
    --skip_tar ${PROC_SKIP:-0} \
    --num_work_tar ${PROC_NUM:-16} \
    --load-8bit 0 \
    --load-4bit 0 \
    --process ${PROC_WORKER:-16} \
    --global_input_image_num ${PROC_GLOBAL:-"8"} \
    --num_global_prompts 1 \