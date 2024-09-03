import os
import re
import argparse
import importlib.util
import subprocess
from tqdm import tqdm
import glob
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from pathlib import Path
import json
import jsonlines
from json_utils import load_json, load_jsonl, save_jsonl
from multiprocessing import Pool, set_start_method

import os

configs = {
    "math23k": {
        "answer_column": "ans",
    },
    "mathqa": {
        "answer_column": "correct",
    },
    "ape210k": {
        "answer_column": "ans",
    },
    "gsm8k": {
        "answer_column": "direct_answer",
    },
    "mmlu": {
        "answer_column": "target",
    },
    "bb_arithmetics": {
        "answer_column": "target",
    },
    "arith_std": {
        "answer_column": "answer",
    },
    "gaokao": {
        "answer_column": "answer",
    },
    "mawps": {
        "answer_column": "ans",
    },
    "bbh": {
        "answer_column": "target",
    },
    "scq_en": {
        "answer_column": "answer",
    },
    "scq_ch": {
        "answer_column": "answer",
    },
    "math": {
        "answer_column": "solution",
    },
    "asdiv": {
        "answer_column": "answer",
    },
    "svamp": {
        "answer_column": "answer",
    },
    "math401": {
        "answer_column": "response",
    },
    "draw": {
        "answer_column": "answer",
    },
    "dolphin1878": {
        "answer_column": "ans",
    },
    "hmwp": {
        "answer_column": "ans",
    },
    "ceval": {
        "answer_column": "answer",
    },
    "agieval": {
        "answer_column": "label",
    },
    "cmmlu": {
        "answer_column": "Answer",
    },
    "olypiadbench": {
        "answer_column": "answer",
    },
}

chat_prompt = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>human
{}<|im_end|>
<|im_start|>gpt
"""

def build_user_query(question, pred_answer, answer):
    instruct = "作为答案校验者，你将处理一个包括“数学问题”、“解答”及“模型预测结果”的数据结构。你的工作是从“解答”和“模型预测”部分中精确抽取数学问题的每一步答案。然后，仔细比较这两组答案的每个对应步骤。如果所有子问题的答案在意义上完全匹配，你应该在最后返回<answer>correct</answer>。反之，如果有任何不匹配之处，你应返回<answer>incorrect</answer>。务必逐步分析并清晰阐述你的对比逻辑。"
    value_human = f"{instruct}\n\n数学问题:\n{question}\n\n解答:{answer}\n\n模型推理结果:\n{pred_answer}"
    
    input_text = chat_prompt.format(value_human)
    return input_text

def build_input_data(row, datafile):
    data_name_with_shot = Path(datafile).name

    one_data_name_row = data_name_with_shot.split("-")[0].lower()
    config = configs[one_data_name_row]

    row_content_meta = row
    row_meta_meta = row_content_meta["meta"]
    
    user_query = build_user_query(
        row_content_meta["conversations"][-1]["value"],
        row_content_meta["raw_response"],
        row_meta_meta.get(config["answer_column"], None)
    )
    return user_query

def process_data_with_chat_responses(data, model, tokenizer, device, data_file, args):
    processed_data = []
    data_file_basename = os.path.basename(data_file)
    for idx, item in tqdm(enumerate(data), total=len(data), desc=f"Processing data {data_file_basename}"):
        prompt = build_input_data(item, data_file)
        model_inputs = tokenizer([prompt], return_tensors="pt", padding=True).to(device)

        # 获取 input_ids 和 attention_mask
        input_ids = model_inputs.input_ids
        attention_mask = model_inputs.attention_mask

        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            temperature=0.01,
            max_new_tokens=16,
            eos_token_id=100005,
            pad_token_id=tokenizer.pad_token_id
        )

        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
        item["raw_response"] = response
        item["original_prompt"] = prompt
        processed_data.append(item)
    return processed_data

def generate_chat_responses(model, tokenizer, data_file, output_file, device, args):
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    data = load_jsonl(data_file)
    data_input = data
    
    print("Number of samples:", len(data_input))
    processed_data = process_data_with_chat_responses(data_input, model, tokenizer, device, data_file, args)
    save_jsonl(output_file, processed_data)

def process_file(file_info):
    model_path, one_data_info, output_file, device, args = file_info
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    print(f"Processing file: {one_data_info}")
    generate_chat_responses(model, tokenizer, one_data_info, output_file, device, args)

def generate_chat_responses_all(args):
    all_data_info = glob.glob(os.path.join(args.input_dir, '**', '*.json'), recursive=True)
    
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    print("Device to inference: {}".format(device))
    model_path = args.model_path
    
    file_info_list = [(model_path, one_data_info, os.path.join(args.output_dir, os.path.basename(one_data_info)), device, args) for one_data_info in all_data_info]
    
    with Pool(processes=args.device_num) as pool:
        pool.map(process_file, file_info_list)

def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created directory: {args.output_dir}")
    else:
        print(f"Directory already exists: {args.output_dir}")
    
    generate_chat_responses_all(args)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/mnt/pfs/zitao_team/big_model/raw_models/DeepSeek-7B-Math-Compare-Answer")
    parser.add_argument("--input_dir", type=str, default="/mnt/pfs/zitao_team/fangzhensheng/matheval-new/MathEval-new/result_nips_vllm/0510/llama-3-8b-rewrite")
    parser.add_argument("--output_dir", type=str, help="Path to the output file", default="/mnt/pfs/zitao_team/fangzhensheng/MathEval/compare_result/llama-3-8b-rewrite")
    parser.add_argument(
        "--device_num", type=int, default=4, help="number of gpus to use"
    )   
    # Set the start method to 'spawn'
    set_start_method('spawn', force=True)

    args = parser.parse_args()
    main(args)
