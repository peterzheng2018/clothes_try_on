import argparse
import os
import os.path as osp

import cv2
from diffusers import AutoencoderKL, StableDiffusionClothPipeline, UNet2DConditionModel
import numpy as np
import torch
from transformers import CLIPTextModel, CLIPTokenizer

from train_fashin_2019 import ClothDataset, collate_fn

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script for InstructPix2Pix.")
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--save_data_dir",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    return args
    

def main():
    args = parse_args()
    model_id = args.model_id
    save_model = args.model_checkpoint
    train_data_dir = args.train_data_dir
    out_dir = args.save_data_dir
    num_validation_images = 5
    
    if not osp.exists(out_dir):
        os.makedirs(out_dir)
    
    tokenizer = CLIPTokenizer.from_pretrained(
        model_id, subfolder="tokenizer", revision=None
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", revision=None
    )
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", revision=None)
    unet = UNet2DConditionModel.from_pretrained(
        save_model, subfolder="unet", revision=None
    )
    
    pipe = StableDiffusionClothPipeline.from_pretrained(model_id, unet=unet, text_encoder=text_encoder, vae=vae, torch_dtype=torch.float16).to("cuda")
    
    val_dataset = ClothDataset(
        instance_data_root=train_data_dir,
        instance_mode='val',
        instance_file='train.csv',
        label_descriptions_file='label_descriptions.json',
        tokenizer=tokenizer,
        size=256,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=lambda examples: collate_fn(examples, "val", False),
        batch_size=1,
        num_workers=1,
    )
    
    generator = None
    image_logs = []
    for step, batch in enumerate(val_dataloader):
        original_image = batch["pixel_values"]
        prompt_mask = batch["prompt_mask"]
        target_values = batch["edit_values"]
        validation_prompt = batch["ori_text_prompt_mask"]

        images = []
        for _ in range(num_validation_images):
            with torch.autocast("cuda"):
                image = pipe(
                    validation_prompt,
                    image=original_image,
                    prompt_mask=prompt_mask,
                    num_inference_steps=20,
                    image_guidance_scale=1.5,
                    guidance_scale=7,
                    generator=generator
                ).images[0]

            images.append(image)

        image_logs.append(
            {"validation_image": batch["ori_edit_values"][0], "prompt_mask": batch["ori_prompt_mask"][0], "images": images}
        )
        
    for i, log in enumerate(image_logs):
        images = log["images"]
        prompt_mask = log["prompt_mask"]
        validation_image = log["validation_image"]
        
        formatted_images = []

        formatted_images.append(np.asarray(validation_image))
        formatted_images.append(np.asarray(prompt_mask))

        for image in images:
            formatted_images.append(np.asarray(image))
            
        formatted_images = np.hstack(formatted_images)
        
        cv2.imwrite(f"{out_dir}/image_{prompt_mask}_{i}.png", formatted_images)
