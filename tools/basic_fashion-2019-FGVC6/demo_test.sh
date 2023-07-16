export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="/data/rlan/data/imaterialist-fashion-2019-FGVC6"
export MODEL_PATH="instruct-pix2pix-model/checkpoint-15000"
export SAVE_DATA_DIR="results/"

python test_pipeline.py \
    --model_id $MODEL_NAME \
    --model_checkpoint $MODEL_PATH \
    --train_data_dir $DATA_DIR \
    --save_data_dir $SAVE_DATA_DIR
