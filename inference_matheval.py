import abc
import math
from typing import Iterable, Optional
import sys
import warnings
import os
import copy
import transformers
import torch
from prompt_builder import build_prompt
from json_utils import load_json, load_jsonl, save_jsonl
from transformers import AutoTokenizer
from generate_shell_config import root_model_info, all_data_info

base_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(base_path, "../"))

from tqdm import tqdm
import pandas as pd
import argparse



model_info_dict = {x['model_name']:x for x in root_model_info}

def generate_one_response(model, tokenizer, device, prompt, args):
    system_template = """{}"""
    params = {
        "prompt": system_template.format(prompt),
        "temperature": 0.01,
        "top_p": 1.0,
        "max_new_tokens": args.max_new_tokens,
        "stream_interval": 1,
    }
    completion = generate_stream(model, tokenizer, params, device)
    
    response = None
    # Iterate through the completion generator
    for one_text in completion:
        response = one_text  # Update response with the latest generated text
    
    return response

def process_data_with_chat_responses(data, model, tokenizer, device, model_inference_config, args):
    processed_data = []
    inference_mode = "few-shot" if args.few_shot else "zero-shot"
    print(f"[INFO]: Activate {inference_mode} inference.")
    
    template_name = model_inference_config['template_name']
    
    if args.accelerator == 'vllm':
        prompts_list = []
        for item in tqdm(data):
            prompt = build_prompt(item["conversations"], template_name, args)
            prompts_list.append(prompt)
        
        # may change here
        sampling_params = SamplingParams(
            temperature=0.01,
            top_p=0.95,
            repetition_penalty=1.0,
            max_tokens=512,
            # stop_token_ids=[tokenizer.eos_token_id],
        )
        
        response_list = model.generate(prompts_list, sampling_params)
        for idx, item in enumerate(data):
            item["raw_response"] = response_list[idx].outputs[0].text
            item["original_prompt"] = prompts_list[idx]
            processed_data.append(item)
    else:
        for item in tqdm(data):
            prompt = build_prompt(item['conversations'], template_name, args)
            response = generate_one_response(model, tokenizer, device, prompt, args)
            item["original_prompt"] = prompt
            item["raw_response"] = response
            processed_data.append(item)
            print("Raw prompt:", prompt)
            print("Generated chat response:", response)
    
    return processed_data

def generate_chat_responses(model, tokenizer, data_file, output_file, device, args):
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    if data_file.endswith("json"):
        data = load_json(data_file)
    else:
        data = load_jsonl(data_file)
    
    if args.accelerator == 'vllm':
        data_input = data
    else:
        data_input = data[args.start_index:args.end_index] # we slice the original big data to different partitions.
    
    print("Number of samples:", len(data_input))
    model_name = args.model_name # the model name
    model_inference_config = model_info_dict[model_name]
    processed_data = process_data_with_chat_responses(data_input, model, tokenizer, device, model_inference_config, args)
    save_jsonl(output_file, processed_data)
    
def generate_chat_responses_all_vllm(args):
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    print("Device to inference: {}".format(device))
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    model_path = args.model_path
    if args.accelerator == 'vllm':
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if "qwen" in args.model_name.lower() or "internlm" in args.model_name.lower():
            model = LLM(
                model=model_path,
                trust_remote_code=True,
                tensor_parallel_size=args.device_num,
            )
        else:
            model = LLM(model=model_path, tensor_parallel_size=args.device_num)
    
    for one_data_info in all_data_info:
        base_data_name = os.path.basename(one_data_info["data_path"]).replace(".jsonl", "")
        output_file = os.path.join(args.output_dir, base_data_name + ".json")
        generate_chat_responses(model, tokenizer, one_data_info["data_path"], output_file, device, args)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="chatglm2-6b", help="Name of the model"
    )
    parser.add_argument("--model_path", type=str, default="/mnt/pfs/zitao_team/tianqiaoliu/public_github/ChatGLM2-6B/ptuning/output/mathgpt-chatglm2-6b-ft-2e-5/checkpoint-POINTNUM")
    parser.add_argument("--output_dir", type=str, help="Path to the output file", default="./results/chatglm2-6b/test_data_small_with_response_chatglm2_POINTNUM.json")
    parser.add_argument("--start_index", type=int, help="Where to start the slice of the dataset")
    parser.add_argument("--end_index", type=int, help="The size of the slice of the dataset")
    parser.add_argument("--device_num", type=int, default=1, help="number of gpus to use")
    parser.add_argument("--max_new_tokens", type=int, help="The maximum num of max new tokens")
    parser.add_argument("--stop_str", type=str, default="", help="the stop string for the model for this dataset")
    parser.add_argument(
        "--accelerator", 
        type=str, 
        default="vllm", 
        help="Specify the accelerator for inference. Supported options: 'vllm'. Leave empty for default settings."
    )
    parser.add_argument('--few_shot', action='store_true', help='whether to activate few shot or not')
    args = parser.parse_args()
    
    if args.accelerator == 'vllm':
        from vllm import LLM, SamplingParams
        print("generate responses accelerator vllm.")
        generate_chat_responses_all_vllm(args)
        
