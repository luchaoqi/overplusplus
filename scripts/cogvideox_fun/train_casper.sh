export MODEL_NAME="models/Diffusion_Transformer/CogVideoX-Fun-V1.5-5b-InP"
export DATASET_NAME="/playpen-nas-ssd/luchao/data/omnimatte/gen_omnimatte_davis"
export DATASET_META_NAME="/playpen-nas-ssd/luchao/data/omnimatte/gen_omnimatte_davis/casper.json"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/cogvideox_fun/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=1024 \
  --video_sample_size=256 \
  --token_sample_size=512 \
  --video_sample_stride=1 \
  --video_sample_n_frames=85 \
  --train_batch_size=2 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --num_train_epochs=100 \
  --checkpointing_steps=2000 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="cogvideox-casper" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --random_frame_crop \
  --enable_bucket \
  --use_came \
  --use_deepspeed \
  --train_mode="casper" \
  --trainable_modules "." \
  --dataloader_num_workers=0 \
  --resume_from_checkpoint="latest" \
  
export MODEL_NAME="models/Diffusion_Transformer/CogVideoX-Fun-V1.5-5b-InP"
export DATASET_NAME="/playpen-nas-ssd/luchao/data/omnimatte/gen_omnimatte_davis_train"
export DATASET_META_NAME="/playpen-nas-ssd/luchao/data/omnimatte/gen_omnimatte_davis_train/casper.json"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

export MODEL_NAME="models/Diffusion_Transformer/CogVideoX-Fun-V1.5-5b-InP"
export DATASET_NAME="/playpen-nas-ssd/luchao/data/omnimatte/gen_omnimatte_vos"
export DATASET_META_NAME="/playpen-nas-ssd/luchao/data/omnimatte/gen_omnimatte_vos/casper.json"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/cogvideox_fun/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=512 \
  --video_sample_size=256 \
  --token_sample_size=512 \
  --video_sample_stride=3 \
  --video_sample_n_frames=49 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --num_train_epochs=1000 \
  --checkpointing_steps=1000 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="cogvideox-casper-vos" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --random_frame_crop \
  --enable_bucket \
  --use_came \
  --use_deepspeed \
  --train_mode="casper" \
  --dataloader_num_workers=0 \
  --report_to="wandb" \
  --trainable_modules "."
  

export MODEL_NAME="models/Diffusion_Transformer/CogVideoX-Fun-V1.5-5b-InP"
export DATASET_NAME=""
export DATASET_META_NAME="/playpen-nas-ssd/luchao/projects/gen-omnimatte-public/datasets/casper.json"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO


accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/cogvideox_fun/train.py \
  --pretrained_model_name_or_path="/playpen-nas-ssd/luchao/projects/gen-omnimatte-public/models/Diffusion_Transformer/CogVideoX-Fun-V1.5-5b-InP" \
  --train_data_meta="/playpen-nas-ssd/luchao/projects/gen-omnimatte-public/datasets/train-casper/casper.json" \
  --image_sample_size=512 \
  --video_sample_size=256 \
  --token_sample_size=512 \
  --video_sample_stride=1 \
  --video_sample_n_frames=85 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --num_train_epochs=5 \
  --checkpointing_steps=1000 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --random_frame_crop \
  --enable_bucket \
  --use_came \
  --use_deepspeed \
  --train_mode="casper" \
  --dataloader_num_workers=0 \
  --report_to="wandb" \
  --trainable_modules "." \
  --binarize_mask \
  --output_dir="training/cogvideox-casper-gen_omnimatte_dualmask"