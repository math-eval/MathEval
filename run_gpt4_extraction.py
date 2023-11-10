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


def build_user_query(question, analysis):
    return "USER:{}\n\n模型生成的解析: {}".format(question, analysis)

def build_user_query_math(question, analysis):
    analysis_splited = analysis.split("\n\nProblem:")[0]
    return "USER:{}\n\n模型生成的解析: {}".format(question, analysis_splited)

def find_no_extraction(processed_file, file_to_process):
    file_to_fill = []
    processed_questions = []
    for line in processed_file:
        row_content = json.loads(line)

        if not isinstance(row_content["meta"], dict):
            row_content_meta = json.loads(row_content["meta"])
        else:
            row_content_meta = row_content["meta"]
        
        if "conversations" not in row_content_meta:
            if not isinstance(row_content_meta["meta"], dict):
                row_content_meta = json.loads(row_content_meta["meta"])
            else:
                row_content_meta = row_content_meta["meta"]
        
        processed_questions.append(row_content_meta["conversations"][-1]["value"])
    
    for line in file_to_process:
        line_to_process = json.loads(line)
        if "conversations" in line_to_process:
            line_conv = line_to_process["conversations"][-1]['value']
        else:
            if not isinstance(line_to_process["meta"], dict):
                row_content_meta = json.loads(line_to_process["meta"])
            else:
                row_content_meta = line_to_process["meta"]
            line_conv = row_content_meta["conversations"][-1]['value']
        if line_conv not in processed_questions:
            file_to_fill.append(line)
    return file_to_fill

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
    example_files = [x for x in example_files if "examples0.md" in x]
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


prompt_dir = os.path.join(current_folder, "prompts", "extraction_prompts")
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
    row_content = json.loads(row)
    if "raw_response" not in row_content:
        row_content["conversations"] = json.loads(row_content["meta"])["conversations"]
        row_content["raw_response"] = {"gen_text": row_content["response"]}
    user_query = build_user_query(
        row_content["conversations"][-1]["value"],
        row_content["raw_response"]["gen_text"],
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

def build_input_data_math(row):
    """
    根据给定的数据集行，构建输入数据。
    """
    # idx, row = row
    # query = row.to_json()
    row_content = json.loads(row)
    if "raw_response" not in row_content:
        row_content["conversations"] = json.loads(row_content["meta"])["conversations"]
        row_content["raw_response"] = {"gen_text": row_content["response"]}
    user_query = build_user_query_math(
        row_content["conversations"][-1]["value"],
        row_content["raw_response"]["gen_text"],
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


def process_row_math(row):
    try:
        input_data = build_input_data_math(row)
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


# 定义处理数据集的函数
def process_dataset_math(dataset, output_file, n_jobs):
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
        for response in pool.imap_unordered(partial(process_row_math), dataset):
            wrapped_callback(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument('--data_name', type=str)
    args = parser.parse_args()
    debug = False
    n_jobs = 30
    output_dir = ""
    data_to_fill_record = []
    for one_data_info in all_data_info:
        for one_model_info in root_model_info:
            model_name = one_model_info["model_name"]
            model_path = one_model_info["model_path"]
            device_num = one_model_info["num_gpu"]
            data_file = one_data_info["data_path"]
            data_name = one_data_info["data_name"]
            if model_name == args.model_name:
                if data_name == args.data_name:
                    this_model_data_save_dir = os.path.join(
                        output_dir, model_name, data_name
                    )
                    all_json_files = glob.glob(
                        os.path.join(this_model_data_save_dir, "*.json")
                    )
                    print(model_name)
                    print(data_name)

                    for one_json_file in all_json_files:
                        with open(one_json_file, "r") as f:
                            json_content = f.readlines()
                        base_json_file_name = os.path.basename(one_json_file)
                        extraction_output_file = os.path.join(
                            output_dir,
                            model_name,
                            data_name,
                            base_json_file_name.replace(".json", "")
                            + "_gpt4_extraction.jsonl",
                        )
                        if os.path.exists(extraction_output_file):
                            with open(extraction_output_file, "r") as f:
                                extractioned_lines = f.readlines()
                                
                            if len(extractioned_lines) != len(json_content):
                                # This is waited to be processed
                                data_to_fill_record.append((model_name, data_name, len(json_content)-len(extractioned_lines)))
                                print(data_name)
                                print(model_name)
                                if len(json_content)-len(extractioned_lines) >=10:
                                    lines_to_fill = find_no_extraction(extractioned_lines, json_content)
                                    if data_name == "math-4shot":
                                        process_dataset_math(
                                            lines_to_fill, extraction_output_file, n_jobs
                                        )
                                    else:
                                        process_dataset(
                                            lines_to_fill, extraction_output_file, n_jobs
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
                        else:
                            print(data_name)
                            print(model_name)
                            print("No record for this")
                            if data_name == "math-4shot":
                                process_dataset_math(
                                    json_content, extraction_output_file, n_jobs
                                )
                            else:
                                process_dataset(
                                    json_content, extraction_output_file, n_jobs
                                )
                            print(
                                    "Done for model {} data {} partition {}".format(
                                        model_name, data_name, base_json_file_name
                                    )
                                )
                    # print("This is Done")


