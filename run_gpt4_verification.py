import pandas as pd
import json
from tqdm import tqdm
import os
import glob
import random
import multiprocessing as mp
from functools import partial
import requests
import uuid
import time
import hashlib
import json
import argparse
from generate_shell_config import all_data_info, root_model_info
current_folder = os.path.dirname(os.path.abspath(__file__))


# 计算验签值
# 输入参数的类型：字符串
def CalSignature(time_stamp, nonce, device_id):
    sign = sign_base.format(access_key_secret, time_stamp, nonce, device_id)
    md = hashlib.md5()
    md.update(sign.encode())
    return md.hexdigest()


def load_prompts(path):
    with open(path, "r", encoding="utf8") as f:
        return "".join(f.readlines())


def build_user_query(question, real_answer, model_answer):
    return "Math Word Problem:{}\n\nReal Answer:{}\n\nModel-generated Answer:{}".format(
        question, real_answer, model_answer
    )


def process_files(prompt_dir, keep_examples, system_input_md_path):
    system_input = load_prompts(system_input_md_path)
    if len(keep_examples) == 0:
        keep_examples = [
            x for x in os.listdir(prompt_dir) if "example" in x and x[0] != "."
        ]

    example_files = []
    for fname in keep_examples:
        file_path = os.path.join(prompt_dir, fname)
        example_files.append(file_path)
        # api_list.extend(extract_actions_from_path(file_path))

    examples_input = []
    for example_file in example_files:
        with open(example_file, "r") as f:
            example_lines = f.readlines()
        for example_line in example_lines:
            if example_line.startswith("USER"):
                example_line_user = example_line.strip("\n").replace("USER: ", "")
                examples_input.append({"role": "user", "content": example_line_user})
            elif example_line.startswith("ASSISTANT"):
                example_line_user = example_line.strip("\n").replace("ASSISTANT: ", "")
                examples_input.append(
                    {"role": "assistant", "content": example_line_user}
                )

    return system_input, examples_input


prompt_dir = os.path.join(current_folder, "prompts", "verification_prompts")
system_input_info = os.path.join(prompt_dir, "instruction.md")
system_input_global, examples_input_global = process_files(
    prompt_dir, [], system_input_info
)


def build_input_data(row):
    """
    根据给定的数据集行，构建输入数据。
    """
    # idx, row = row
    # query = row.to_json()
    if not isinstance(row, dict):
        row_content = json.loads(row)
    else:
        row_content = row
    
    if not isinstance(row_content["meta"], dict):
        row_content_meta = json.loads(row_content["meta"])
    else:
        row_content_meta = row_content["meta"]
    
    if row_content["gpt4_extraction"] == "":
        generated_answer = row_content["regex_extraction"]
    else:
        generated_answer = row_content["gpt4_extraction"]
    # print(f"This generated answer is {generated_answer}")

    if "conversations" not in row_content_meta:
        if not isinstance(row_content_meta["meta"], dict):
            row_content_meta = json.loads(row_content_meta["meta"])
        else:
            row_content_meta = row_content_meta["meta"]
    user_query = build_user_query(
        row_content_meta["conversations"][-1]["value"],
        row_content["ground_truth"],
        generated_answer,
    )
    # gpt4
    data = {
        "system": system_input_global,
        "examples": examples_input_global,
        "question": user_query,
        "temperature": 0,
        "frequency_penalty": 1,
        "presence_penalty": 1,
        "engine": "GPT4",
        "max_tokens": 4096,
        "max_retry": 20,
    }
    # # wenxin
    # data = {
    #     "message": query,
    # }
    time.sleep(0.3)
    return data


def process_row(row):
    try:
        input_data = build_input_data(row)
        response = send_chat_request(**input_data)
        if response is not None:
            response["meta"] = row
        # response = CallWenXin(input_data)
        return response
    except Exception as e:
        print(f"Error processing row {row}: {e}")
        return None


# 定义处理数据集的函数
def process_dataset(dataset, output_file, n_jobs):
    # n_processes = min(cpu_count(), len(dataset))

    n_processes = n_jobs

    pbar = tqdm(total=len(dataset), desc="Processing files", dynamic_ncols=True)

    def wrapped_callback(response):
        if response is not None:
            with open(output_file, "a") as f:
                json.dump(response, f, ensure_ascii=False)
                f.write("\n")  # 每个响应为一行
        pbar.update(1)

    with mp.Pool(n_processes) as pool:
        for response in pool.imap_unordered(partial(process_row), dataset):
            wrapped_callback(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument('--data_name', type=str)
    parser.add_argument("--output_dir")
    args = parser.parse_args()
    debug = False
    n_jobs = 30
    output_dir = ""
    for one_data_info in all_data_info:
        for one_model_info in root_model_info:
            model_name = one_model_info["model_name"]
            model_path = one_model_info["model_path"]
            device_num = one_model_info["num_gpu"]
            data_file = one_data_info["data_path"]
            data_name = one_data_info["data_name"]
            if data_name not in ["cmmlu-3shot",
                    "cmmlu-0shot",
                    "ceval-0shot",
                    "scq_ch-3shot",
                    "scq_ch-0shot",
                    "scq_en-0shot",
                    "scq_en-3shot",
                    "GAOKAO-BENCH-3shot",
                    "GAOKAO-BENCH-0shot",
                    "mmlu-3shot",
                    "mmlu-0shot",
                    "MathQA-3shot",
                    "MathQA-0shot"]:
                if model_name == args.model_name:
                    # 这些是坤神觉得不好处理的，除了选择题其他的都emo
                    if data_name == args.data_name:
                        print(data_name)
                        print(model_name)
                        this_model_data_save_dir = os.path.join(
                            output_dir, model_name, data_name
                        )
                        # if this_model_data_save_dir in ["/mnt/pfs/zitao_team/tianqiaoliu/mathEval_data_check/result/LLaMa2-7B/scq_en-3shot", "/mnt/pfs/zitao_team/tianqiaoliu/mathEval_data_check/result/LLaMa2-7B/ape210k-0shot"]:
                        all_json_files = glob.glob(
                            os.path.join(this_model_data_save_dir, "*for_judge.jsonl")
                        )

                        for one_json_file in all_json_files:
                            with open(one_json_file, "r") as f:
                                json_content = f.readlines()
                            base_json_file_name = os.path.basename(one_json_file)
                            verification_output_file = os.path.join(
                                output_dir,
                                model_name,
                                data_name,
                                base_json_file_name.replace(".jsonl", "")
                                + "_add_gpt4_verification.jsonl",
                            )
                            if not os.path.exists(verification_output_file):
                                process_dataset(
                                    json_content, verification_output_file, n_jobs
                                )
                                # for one_json_content in tqdm(json_content):
                                #     response = process_row(one_json_content)
                                #     with open(extraction_output_file, "a") as f:
                                #         f.write(json.dumps(response, ensure_ascii=False)+"\n")
                                print(
                                    "Done for model {} data {} partition {}".format(
                                        model_name, data_name, base_json_file_name
                                    )
                                )
                        print("This is Done.")
