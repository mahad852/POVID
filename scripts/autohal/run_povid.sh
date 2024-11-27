python llava/train/train_dpo_inherent.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --model_name_or_path ./checkpoint/output/AutoHal_stage_one_merged \
    --version v1 \
    --data_path /home/mali2/datasets/AutoHal/autohal_spatial_relationship_preference_data_train_mod.json \
    --image_folder /home/mali2/datasets/AutoHal \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 False \
    --output_dir ./checkpoint/output/AutoHal_stage_two_LoRa \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1\
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 1 \
    --learning_rate 1e-7 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to wandb \
    --tf32 False \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
