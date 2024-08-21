import os
import re
import argparse
import importlib.util
import subprocess
from tqdm import tqdm
import glob
import torch
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from pathlib import Path
import json
import jsonlines
from json_utils import load_json, load_jsonl, save_jsonl

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

configs = {
    "math23k": {
        # math23k
        # https://arxiv.org/pdf/2109.03034v1.pdf
        "name": "math23k",
        "answer_column": "ans",
        "shots": [0, 3],
        "options": r"-?\d+\.\d+|-?\d+/?\d*|\d*frac\{-?\d+\}\{\d+\}",
    },
    "mathqa": {
        # MathQA
        # https://arxiv.org/pdf/1907.01642v1.pdf
        "name": "MathQA",
        "answer_column": "correct",
        "shots": [0, 3],
        "options": r"-?\d+\.\d+|-?\d+/?\d*|\d*frac\{-?\d+\}\{\d+\}|[a-e]",
    },
    "ape210k": {
        # ape210k
        # https://github.com/Chenny0808/ape210k
        "name": "ape210k",
        "answer_column": "ans",
        "shots": [0, 3],
        "options": r"-?\d+\.\d+|-?\d+/?\d*|\d*frac\{-?\d+\}\{\d+\}",
    },
    "gsm8k": {
        # GSM8K
        # https://arxiv.org/pdf/2110.14168.pdf
        "name": "GSM8K",
        "answer_column": "direct_answer",
        "shots": [0, 3],
        "options": r"-?\d+\.\d+|-?\d+/?\d*|\d*frac\{-?\d+\}\{\d+\}",
    },
    "mmlu": {
        # mmlu
        # https://arxiv.org/pdf/2009.03300.pdf
        "name": "mmlu",
        "answer_column": "target",
        "shots": [0, 3],
        "options": r"-?\d+\.\d+|-?\d+/?\d*|\d*frac\{-?\d+\}\{\d+\}|[A-F]",
    },
    "bb_arithmetics": {
        # bb_arithmetics
        # https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/arithmetic
        "name": "bb_arithmetics",
        "answer_column": "target",
        "shots": [0, 3],
        "options": r"-?\d+\.\d+|-?\d+/?\d*",
    },
    "arith_std": {
        # arith_std
        # from TAL
        "name": "arith_std",
        "answer_column": "answer",
        "shots": [0, 3],
        "options": r"####\s?(\-?\d+[\.|\,]?\d*i*)\s?|answer is (\-?\d+[\.|\,]?\d*i*)|=\s*(\-?\d+[\.|\,]?\d*i*)\n|(\-?\d+[\.|\,]?\d*i*)\n|(\-?\d+[\.|\,]?\d*i*)",
    },
    "gaokao": {
        # GAOKAO-BENCH
        # https://arxiv.org/pdf/2305.12474v2.pdf
        "name": "GAOKAO-BENCH",
        "answer_column": "answer",
        "shots": [0, 3],
        "options": r"-?\d+\.\d+|-?\d+/?\d*|\d*frac\{-?\d+\}\{\d+\}|[A-F]",
    },
    "mawps": {
        # mawps
        # https://aclanthology.org/N16-1136.pdf
        "name": "mawps",
        "answer_column": "ans",
        "shots": [0, 3],
        "options": r"####\s?(\-?\d+[\.|\,]?\d*)\s?|answer is (\-?\d+[\.|\,]?\d*)|answer is .*\=\s?(\-?\d+[\.|\,]?\d*)\$|\=\s?(\-?\d+[\.|\,]?\d*).*\n|(\-?\d+[\.|\,]?\d*)",
    },
    "bbh": {
        # BBH
        # https://arxiv.org/pdf/2210.09261.pdf
        "name": "BBH",
        "answer_column": "target",
        "shots": [0, 3],
        "options": r"-?\d+\.\d+|-?\d+/?\d*|\d*frac\{-?\d+\}\{\d+\}|[A-F]",
    },
    "scq_en": {
        # scq_en
        # https://huggingface.co/datasets/math-eval/TAL-SCQ5K
        "name": "scq_en",
        "answer_column": "answer",
        "shots": [0, 3],
        "options": r"-?\d+\.\d+|-?\d+/?\d*|\d*frac\{-?\d+\}\{\d+\}|[A-F]",
    },
    "scq_ch": {
        # scq_ch
        # https://huggingface.co/datasets/math-eval/TAL-SCQ5K
        "name": "scq_ch",
        "answer_column": "answer",
        "shots": [0, 3],
        "options": r"-?\d+\.\d+|-?\d+/?\d*|\d*frac\{-?\d+\}\{\d+\}|[A-F]",
    },
    "math": {
        # math
        # https://arxiv.org/pdf/2103.03874.pdf
        "name": "math",
        "answer_column": "solution",
        "shots": [0, 3],
        "options": r"-?\d+\.\d+|-?\d+/?\d*|\d*frac\{-?\d+\}\{\d+\}",
    },
    "asdiv": {
        # asdiv-a
        # https://github.com/LYH-YF/MWPToolkit/blob/master/dataset/asdiv-a/
        "name": "asdiv-a",
        "answer_column": "answer",
        "shots": [0, 3],
        "options": r"-?\d+\.\d+|-?\d+/?\d*|\d*frac\{-?\d+\}\{\d+\}|[A-F]",
    },
    "svamp": {
        # svamp
        # https://github.com/arkilpatel/SVAMP
        "name": "svamp",
        "answer_column": "answer",
        "shots": [0, 3],
        "options": r"-?\d+\.\d+|-?\d+/?\d*|\d*frac\{-?\d+\}\{\d+\}|[A-F]",
    },
    "math401": {
        # math401
        # https://github.com/GanjinZero/math401-llm
        "name": "math401",
        "answer_column": "response",
        "shots": [0, 3],
        "options": r"-?\d+\.\d+|-?\d+/?\d*|\d*frac\{-?\d+\}\{\d+\}|[A-F]",
    },
    "draw": {
        # draw
        # https://github.com/LYH-YF/MWPToolkit/blob/master/dataset/draw/
        "name": "draw",
        "answer_column": "answer",
        "shots": [0, 3],
        "options": r"-?\d+\.\d+|-?\d+/?\d*|\d*frac\{-?\d+\}\{\d+\}|[A-F]",
    },
    "dolphin1878": {
        # dolphin1878
        # https://www.microsoft.com/en-us/research/project/sigmadolphin/
        "name": "dolphin1878",
        "answer_column": "ans",
        "shots": [0, 3],
        "options": r"[-+]?\d+(?:,\d+)?(?:\.\d+)?",
    },
    "hmwp": {
        # hmwp
        # https://github.com/QinJinghui/SAU-Solver
        "name": "hmwp",
        "answer_column": "ans",
        "shots": [0, 3],
        "options": r"-?\d+\.\d+|-?\d+/?\d*|\d*frac\{-?\d+\}\{\d+\}|[A-F]",
    },
    "ceval": {
        # ceval
        # https://github.com/SJTU-LIT/ceval
        "name": "ceval",
        "answer_column": "answer",
        "shots": [0],
        "options": r"-?\d+\.\d+|-?\d+/?\d*|\d*frac\{-?\d+\}\{\d+\}|[A-F]",
    },
    "agieval": {
        # AGIEval
        # https://github.com/microsoft/AGIEval/blob/main/
        "name": "AGIEval",
        "answer_column": "label",
        "shots": [0, 3],
        "options": r"-?\d+\.\d+|-?\d+/?\d*|\d*frac\{-?\d+\}\{\d+\}|[A-F\s]{1,4}",
    },
    "cmmlu": {
        # cmmlu
        # https://github.com/haonan-li/CMMLU/blob/master
        "name": "cmmlu",
        "answer_column": "Answer",
        "shots": [0, 3],
        "options": r"-?\d+\.\d+|-?\d+/?\d*|\d*frac\{-?\d+\}\{\d+\}|[A-F]",
    },
}

chat_prompt = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>human
{}<|im_end|>
<|im_start|>gpt
"""

basic_prompt = """## 任务描述\n\n你是一个数学老师，学生提交了题目的解题步骤，你需要参考`题干`，`解析`和`答案`，判断`学生解题步骤`的结果是否正确。忽略`学生解题步骤`中的错误，只关注最后的答案。答案可能出现在`解析`中，也可能出现在`答案`中。\n\n## 输入内容\n\n题干:\n\n```\n{{question}}\n```\n\n解析:\n\n```\n{{analysis}}\n\n```\n\n答案:\n\n```\n{{answer}}\n```\n\n学生解题步骤:\n\n```\n{{pred_step}}\n```\n\n输出:"""
base_prompt = chat_prompt.format(basic_prompt)

def build_user_query(question, pred_answer, answer):
    input_text = base_prompt.replace("{{question}}", question)
    input_text = input_text.replace("{{pred_step}}", pred_answer)
    input_text = input_text.replace("{{answer}}", answer)
    return input_text

  
def build_input_data(row, datafile):
    data_name_with_shot = Path(datafile).parent.name

    one_data_name_row = data_name_with_shot.split("-")[0].lower()
    config = configs[one_data_name_row]

    row_content_meta = row
    row_meta_meta = row_content_meta["meta"]
    
    user_query = build_user_query(
        row_content_meta["conversations"][-1]["value"],
        row_content_meta["raw_response"]['gen_text'],
        row_meta_meta.get(config["answer_column"], None)
        
    )
    return user_query


def process_data_with_chat_responses(data, model, tokenizer, device, data_file, args):
    processed_data = []
    
    if args.accelerator == 'vllm':
        prompts_list = []
        for item in tqdm(data):
            prompt = build_input_data(item, data_file)
            prompts_list.append(prompt)

        # may change here
        sampling_params = SamplingParams(
            temperature=0.01,
            top_p=0.95,
            repetition_penalty=1.0,
            max_tokens=1024,
            stop_token_ids= [100005],
        )
        
        response_list = model.generate(prompts_list, sampling_params)
        print(f"模型回复: {response_list}\n")
        for idx, item in enumerate(data):
            item["raw_response"] = response_list[idx].outputs[0].text
            item["original_prompt"] = prompts_list[idx]
            processed_data.append(item)
    
    return processed_data

def generate_chat_responses(model, tokenizer, data_file, output_file, device, args):
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    #if data_file.endswith("json"):
    #    data = load_json(data_file)
    #else:
    data = load_jsonl(data_file)
    
    if args.accelerator == 'vllm':
        data_input = data
    
    print("Number of samples:", len(data_input))
    processed_data = process_data_with_chat_responses(data_input, model, tokenizer, device, data_file, args)
    save_jsonl(output_file, processed_data)
    
    
def generate_chat_responses_all(args):
    all_data_info = glob.glob(os.path.join(args.input_dir, '**', '*.json'), recursive=True)
    
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    print("Device to inference: {}".format(device))
    model_path = args.model_path
    if args.accelerator == 'vllm':
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = LLM(model=model_path, trust_remote_code=True, tensor_parallel_size=args.device_num)
        print("loading model success.....")
    for one_data_info in all_data_info:
        base_data_name = os.path.basename(one_data_info)
        output_file = os.path.join(args.output_dir, base_data_name + ".json")
        generate_chat_responses(model, tokenizer, one_data_info, output_file, device, args)
        

def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created directory: {args.output_dir}")
    else:
        print(f"Directory already exists: {args.output_dir}")
    
    generate_chat_responses_all(args)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Tianqiao/DeepSeek-7B-Math-Compare-Answer")
    #parser.add_argument("--
    # ", type=str, default="/mnt/pfs/zitao_team/big_model/raw_models/qwen2/Qwen2-72B-Instruct")
    parser.add_argument("--input_dir", type=str, default="/mnt/pfs/zitao_team/fangzhensheng/MathEval/result_cot/Qwen-14B-Chat")
    parser.add_argument("--output_dir", type=str, help="Path to the output file", default="/mnt/pfs/zitao_team/fangzhensheng/MathEval/compare_result/Qwen-14B-Chat")
    parser.add_argument(
        "--device_num", type=int, default=4, help="number of gpus to use"
    )   
    parser.add_argument(
        "--accelerator", 
        type=str, 
        default="vllm", 
        help="Specify the accelerator for inference. Supported options: 'vllm'. Leave empty for default settings."
    )

    args = parser.parse_args()
    main(args)