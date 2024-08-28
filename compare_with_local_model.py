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
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"


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

def build_user_query(question, pred_answer, answer):
    instruct = "作为答案校验者，你将处理一个包括“数学问题”、“解答”及“模型预测结果”的数据结构。你的工作是从“解答”和“模型预测”部分中精确抽取数学问题的每一步答案。然后，仔细比较这两组答案的每个对应步骤。如果所有子问题的答案在意义上完全匹配，你应该在最后返回<answer>correct</answer>。反之，如果有任何不匹配之处，你应返回<answer>incorrect</answer>。务必逐步分析并清晰阐述你的对比逻辑。"
    value_human = f"{instruct}\n\n数学问题:\n{question}\n\n解答:{answer}\n\n模型推理结果:\n{pred_answer}"
    
    input_text = chat_prompt.format(value_human)
    return input_text

def build_input_data(row, datafile):
    data_name_with_shot = Path(datafile).parent.name

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
    for idx, item in enumerate(data):
        prompt = build_input_data(item, data_file)
        model_inputs = tokenizer([prompt], return_tensors="pt").to(device)

        generated_ids = model.generate(model_inputs.input_ids, temperature=0.01, max_new_tokens=16, eos_token_id=100005)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
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
