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

from torch.utils.data import Dataset

if torch.cuda.is_available():
    device = torch.device("cuda")

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

class HalBench(Dataset):
    def __init__(self, file_path, images_dir):
        with open(file_path, "r") as f:
            self.data = json.load(f)
        
        self.images_dir = images_dir

    def is_visual_input_present(self, item) -> bool:
        return int(item["visual_input"]) != 0

    def __getitem__(self, index):
        item = self.data[index]

        img_path = None
        if self.is_visual_input_present(item):
            img_path = os.path.join(self.images_dir, '/'.join(item["filename"].split('/')[1:]))
        
        prompt = item["question"]

        return item, img_path, prompt

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print(model_name)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, device_map="cpu")
    model = model.to(device)

    input_dir = args.input_dir
    test_file = args.test_file

    dataset = HalBench(test_file, input_dir)

    output_file = args.output_file

    results = []

    with torch.no_grad():
        for (item, file_path, qs) in tqdm(dataset):
            if file_path and not file_path.endswith((".jpg", ".jpeg", ".png")): continue # and nu <= 0:
            
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            elif file_path is not None:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
                
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            if file_path is not None:
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            else:
                input_ids = torch.tensor(tokenizer(prompt).input_ids, dtype=torch.long).unsqueeze(0).cuda()

            image = Image.open(file_path) if file_path is not None else None
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].to(device) if image is not None else None

            with torch.inference_mode():
                output_ids = model.generate(
                    inputs=input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda() if image_tensor else None,
                    do_sample=True,
                    temperature=float(args.temperature),
                    top_p= 1,
                    num_beams= 1,
                    output_attentions=True,
                    # no_repeat_ngram_size=3, args.top_p args.num_beams
                    max_new_tokens=1024,
                    use_cache=True)

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            outputs = outputs.strip()            
            item["model_prediction"] = outputs

            results.append(item)


        with open(output_file, "w") as f:
            json.dump(results, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--test_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--conv-mode", type=str, default="v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)




