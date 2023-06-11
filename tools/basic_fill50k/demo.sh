export CUDA_VISIBLE_DEVICES=0

# use in 11GB GPU
function demo_11GB(){
    export MODEL_DIR="runwayml/stable-diffusion-v1-5"
    export OUTPUT_DIR="output_model"

    accelerate launch train_controlnet.py \
    --pretrained_model_name_or_path=$MODEL_DIR \
    --output_dir=$OUTPUT_DIR \
    --dataset_name=fusing/fill50k \
    --resolution=320 \
    --learning_rate=2e-5 \
    --validation_image "./fusing/conditioning_image_1.png" "./fusing/conditioning_image_2.png" \
    --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --gradient_checkpointing \
    --use_8bit_adam \
    --enable_xformers_memory_efficient_attention \
    --set_grads_to_none \
    --tracker_project_name="controlnet-demo"
}


# use in 16GB GPU
function demo_16GB(){
    export MODEL_DIR="runwayml/stable-diffusion-v1-5"
    export OUTPUT_DIR="output_model"

    accelerate launch train_controlnet.py \
    --pretrained_model_name_or_path=$MODEL_DIR \
    --output_dir=$OUTPUT_DIR \
    --dataset_name=fusing/fill50k \
    --resolution=512 \
    --learning_rate=1e-5 \
    --validation_image "./fusing/conditioning_image_1.png" "./fusing/conditioning_image_2.png" \
    --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --gradient_checkpointing \
    --use_8bit_adam \
    --tracker_project_name="controlnet-demo"
}

# use in normal
function demo(){
    export MODEL_DIR="runwayml/stable-diffusion-v1-5"
    export OUTPUT_DIR="output_model"

    accelerate launch train_controlnet.py \
    --pretrained_model_name_or_path=$MODEL_DIR \
    --output_dir=$OUTPUT_DIR \
    --dataset_name=fusing/fill50k \
    --resolution=512 \
    --learning_rate=1e-5 \
    --validation_image "./fusing/conditioning_image_1.png" "./fusing/conditioning_image_2.png" \
    --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
    --train_batch_size=4 \
    --tracker_project_name="controlnet-demo"
}
