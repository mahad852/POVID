import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava import conversation as conversation_lib
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.model import *

from PIL import Image
import math
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda")

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def get_img_id_to_path_dict(input_dir):
    instances_val = os.path.join(input_dir, "annotations", "instances_val2014.json")
    instances_train = os.path.join(input_dir, "annotations", "instances_train2014.json")
    
    captions_val = os.path.join(input_dir, "annotations", "captions_val2014.json")
    captions_train = os.path.join(input_dir, "annotations", "captions_train2014.json")

    val_img_dir = os.path.join(input_dir, "val2014")
    train_img_dir = os.path.join(input_dir, "train2014")

    if not os.path.exists(instances_val) or not os.path.exists(instances_train) or not os.path.exists(captions_train) or not os.path.exists(captions_val):
        raise Exception("Make sure input_dir to coco is correct. It must have an annotations folder containing instances_val2014.json, instances_train2014.json, captions_val2014.json, captions_train2014.json")
    
    if not os.path.exists(val_img_dir) or not os.path.exists(train_img_dir):
        raise Exception("Make sure input_dir to coco is correct. It must have the following folders containing images: train2014 and val2014")

    img_id_to_path = {}

    for file in [instances_val, captions_val]:#, instances_train, captions_train]:
        image_dicts = json.load(open(file, "r"))["images"]
        img_dir = train_img_dir if "train" in file else val_img_dir
            
        for image_dict in image_dicts:
            img_id_to_path[image_dict["id"]] = os.path.join(img_dir, image_dict["file_name"])
    
    return img_id_to_path


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print(model_name)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, device_map='cpu')
    model = model.to(device)

    input_dir = args.input_dir
    output_file = args.output_file

    img_id_to_path = get_img_id_to_path_dict(input_dir)

    nu = -10000
    with torch.no_grad():
        with open(output_file, "a+") as f:
            for image_id in tqdm(list(img_id_to_path.keys())):
                file_path = img_id_to_path[image_id]
                if file_path.endswith((".jpg", ".jpeg", ".png")) and nu <= 0:
                    if file_path in open(output_file).read():continue
                    qs = 'Describe this image.'
                    cur_prompt = qs
                    if model.config.mm_use_im_start_end:
                        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
                    else:
                        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
                    conv = conv_templates[args.conv_mode].copy()
                    conv.append_message(conv.roles[0], qs)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()
                    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                    image = Image.open(file_path)
                    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].to(device)
                    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                    keywords = [stop_str]
                    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
                    with torch.inference_mode():
                        output_ids = model.generate(
                            inputs=input_ids,
                            images=image_tensor.unsqueeze(0).half().cuda(),
                            do_sample=True,
                            temperature=args.temperature,
                            top_p= 1,
                            num_beams= 1,
                            output_attentions=True,
                            # no_repeat_ngram_size=3, args.top_p args.num_beams
                            max_new_tokens=1024,
                            use_cache=True)
                    input_token_len = input_ids.shape[1]
                    # n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
                    # if n_diff_input_output > 0:
                    #     print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
                    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                    outputs = outputs.strip()
                    result = {"image_id": image_id, "question": cur_prompt, "caption": outputs, "model": "llava_lora_05_05_step_500", "image_name" : file_path.split("/")[-1]} 
                    json.dump(result, f)
                    f.write('\n')
                    f.flush()
                    nu += 1
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="[your final stage lora ckpt path]")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--input_dir", type=str, default="./data/coco")
    parser.add_argument("--output_file", type=str, default="[your output path]")
    parser.add_argument("--conv-mode", type=str, default="v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)




