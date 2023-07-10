export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="/data/rlan/data/imaterialist-fashion-2019-FGVC6"
export RESUME_MODEL="latest"
python train_fashion_2019.py \
    --pretrained_model_name_or_path $MODEL_NAME \
    --resolution 256 --random_flip \
    --train_batch_size 1 --gradient_accumulation_steps 16 --gradient_checkpointing \
    --max_train_steps 15000 \
    --checkpointing_steps 1000 --checkpoints_total_limit 1 \
    --learning_rate 1e-05 --max_grad_norm=1 --lr_warmup_steps=0 \
    --conditioning_dropout_prob=0.05 \
    --use_8bit_adam \
    --enable_xformers_memory_efficient_attention \
    --set_grads_to_none \
    --seed=42 \
    --train_data_dir $DATA_DIR \
    --resume_from_checkpoint $RESUME_MODEL

:'
function demo_11GB(){
    python train_fashion_2019.py \
        --pretrained_model_name_or_path $MODEL_NAME \
        --resolution 256 --random_flip \
        --train_batch_size 1 --gradient_accumulation_steps 16 --gradient_checkpointing \
        --max_train_steps 15000 \
        --checkpointing_steps 5000 --checkpoints_total_limit 1 \
        --learning_rate 1e-05 --max_grad_norm=1 --lr_warmup_steps=0 \
        --conditioning_dropout_prob=0.05 \
        --use_8bit_adam \
        --enable_xformers_memory_efficient_attention \
        --set_grads_to_none \
        --seed=42 \
        --train_data_dir $DATA_DIR
}

function demo_16GB(){
    accelerate launch train_fashion_2019.py \
        --pretrained_model_name_or_path $MODEL_NAME \
        --resolution 256 --random_flip \
        --train_batch_size 4 --gradient_accumulation_steps 4 --gradient_checkpointing \
        --max_train_steps 15000 \
        --checkpointing_steps 5000 --checkpoints_total_limit 1 \
        --learning_rate 1e-05 --max_grad_norm=1 --lr_warmup_steps=0 \
        --conditioning_dropout_prob=0.05 \
        --use_8bit_adam \
        --enable_xformers_memory_efficient_attention \
        --set_grads_to_none \
        --seed=42 \
        --train_data_dir $DATA_DIR
}

# use in normal
function demo(){
    accelerate launch train_fashion_2019.py \
        --pretrained_model_name_or_path $MODEL_NAME \
        --resolution 256 --random_flip \
        --train_batch_size 4 --gradient_accumulation_steps 4 --gradient_checkpointing \
        --max_train_steps 15000 \
        --checkpointing_steps 5000 --checkpoints_total_limit 1 \
        --learning_rate 1e-05 --max_grad_norm=1 --lr_warmup_steps=0 \
        --conditioning_dropout_prob=0.05 \
        --mixed_precision=fp16 \
        --seed=42 \
        --train_data_dir $DATA_DIR
}

function test(){
    model_path=""
    save_data_dir=""
    python test_pipeline.py \
        --model_id $MODEL_NAME \
        --model_checkpoint $model_path \
        --train_data_dir $DATA_DIR \
        --save_data_dir $save_data_dir
}
'
