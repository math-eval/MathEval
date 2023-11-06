"""Inference for FastChat models."""
# from fastchat.train.llama2_flash_attn_monkey_patch import (
#     replace_llama_attn_with_flash_attn,
# )

# replace_llama_attn_with_flash_attn()

import abc
import gc
import math
from typing import Iterable, Optional
import sys
import warnings
import os
import copy
import transformers
from fastchat.model.model_adapter import load_model
from conversation import get_conv_template

base_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(base_path, "../"))
import json
import jsonlines

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoModel,
    AutoModelForSeq2SeqLM,
    T5Tokenizer,
    AutoConfig,
)
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from tqdm import tqdm
import pandas as pd
import argparse
from fastchat.utils import is_partial_stop
from generate_shell_config import root_model_info

model_info_dict = {x['model_name']:x for x in root_model_info}

def prepare_logits_processor(
    temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op so we skip two cases.
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


def find_nearest_thought(generated_text):
    # Split the text at each occurrence of "<thought>"
    segments = generated_text.split("<thought>")
    
    # Return the last segment (which is the content after the last occurrence of "<thought>")
    return "<thought>" + segments[-1]


@torch.inference_mode()
def generate_stream(
    model, tokenizer, params, device, context_len=2048, stream_interval=2, current_prefix_with_api=5, bagging_times=5
):
    print("Generating stream...")
    print(f"Params: {params}")
    use_plugin = params.get("use_plugin", True)
    prompt = params["prompt"]
    len_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", -1))  # -1 means disable
    max_new_tokens = int(params.get("max_new_tokens", 256))
    stop_str = params.get("stop", None)
    echo = bool(params.get("echo", True))
    stop_token_ids = params.get("stop_token_ids", None) or []
    stop_token_ids.append(tokenizer.eos_token_id)

    logits_processor = prepare_logits_processor(
        temperature, repetition_penalty, top_p, top_k
    )

    input_ids = tokenizer(prompt).input_ids
    input_echo_len = len(input_ids)
    output_ids = list(input_ids)
    max_src_len = context_len - max_new_tokens - 8

    input_ids = input_ids[-max_src_len:]

    past_key_values = out = None
    gen_tokens = []
    for i in range(max_new_tokens):
        # print(f"Input ids: {tokenizer.decode(input_ids)}")
        if i == 0 or restart_gen:
            out = model(torch.as_tensor([input_ids], device=device), use_cache=True)
            logits = out.logits
            past_key_values = out.past_key_values
            restart_gen = False
        else:
            out = model(
                input_ids=torch.as_tensor([[token]], device=device),
                use_cache=True,
                past_key_values=past_key_values,
            )
            logits = out.logits
            past_key_values = out.past_key_values

        if logits_processor:
            if repetition_penalty > 1.0:
                tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
            else:
                tmp_output_ids = None
            last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]
        else:
            last_token_logits = logits[0, -1, :]

        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-5 or top_p < 1e-8:  # greedy
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))
                
        gen_tokens.append(token)
        output_ids.append(token)

        if token in stop_token_ids:
            stopped = True
        else:
            stopped = False

        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            if echo:
                tmp_output_ids = output_ids
                rfind_start = len_prompt
            else:
                tmp_output_ids = output_ids[input_echo_len:]
                rfind_start = 0

            output = tokenizer.decode(
                tmp_output_ids,
                spaces_between_special_tokens=False,
            )
            if stop_str:
                if isinstance(stop_str, str):
                    pos = output.rfind(stop_str, rfind_start)
                    if pos != -1:
                        output = output[:pos]
                        stopped = True
                elif isinstance(stop_str, Iterable):
                    for each_stop in stop_str:
                        pos = output.rfind(each_stop, rfind_start)
                        if pos != -1:
                            output = output[:pos]
                            stopped = True
                            break
                else:
                    raise ValueError("Invalid stop field type.")

            yield {
                "text": output,
                "usage": {
                    "prompt_tokens": input_echo_len,
                    "completion_tokens": i,
                    "total_tokens": input_echo_len + i,
                },
                "finish_reason": None,
            }

        if stopped:
            break

    # Finish stream event, which contains finish reason
    if i == max_new_tokens - 1:
        finish_reason = "length"
    elif stopped:
        finish_reason = "stop"
    else:
        finish_reason = None

    yield {
        "text": output,
        "gen_text": tokenizer.decode(gen_tokens,spaces_between_special_tokens=False),
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": i,
            "total_tokens": input_echo_len + i,
        },
        "finish_reason": finish_reason,
    }

    # Clean
    del past_key_values, out
    gc.collect()
    torch.cuda.empty_cache()

def load_json(file_path):
    print("file path here: ", file_path)
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

def load_jsonl(file_path):
    print("file path here: ", file_path)
    with open(file_path, "r") as file:
        raw_data = file.readlines()
    data = []
    for line in raw_data:
        data.append(json.loads(line))
    return data

def save_jsonl(file_path, data):
    # 写入JSONL文件
    with jsonlines.open(file_path, mode="w") as writer:
        writer.write_all(data)


def generate_chat_response(model, tokenizer, device, prompt, args):
    system_template = """{}"""
    params = {
        "prompt": system_template.format(prompt),
        "temperature": 0.01,
        "top_p": 1.0,
        "max_new_tokens": args.max_new_tokens,
        "stream_interval": 1,
    }
    completion = generate_stream(model, tokenizer, params, device)
    for one_text in completion:
        pass
    return one_text


def build_prompt(input_conversation, template_name, args):
    conv = get_conv_template(template_name)
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    
    # Apply prompt templates
    conversations = []
    if roles[input_conversation[0]["from"]] != conv.roles[0]:
        input_conversation = input_conversation[1:]
    
    conv.messages = []
    for j, sentence in enumerate(input_conversation):
        role = roles[sentence['from']]
        assert role == conv.roles[j % 2]
        conv.append_message(role, sentence['value'])
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def process_data_with_chat_responses(data, model, tokenizer, device, model_inference_config, args):
    processed_data = []
    if args.few_shot:
        print("[INFO]: Activate few shot inference.")
    else:
        print("[INFO]: Now is zero-shot settings.")
    template_name = model_inference_config['template_name']
    for item in tqdm(data):
        prompt = build_prompt(item['conversations'], template_name, args)
        response = generate_chat_response(model, tokenizer, device, prompt, args)
        item["original_prompt"] = prompt
        item["raw_response"] = response
        processed_data.append(item)
        print("Raw prompt:", prompt)
        # print("Raw answer:", item["output"])
        print("Generated chat response:", response)
    return processed_data

def generate_chat_responses(model_path, data_file, output_file, args):
    # output dir is
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    print("Device to inference: {}".format(device))
    model, tokenizer = load_model(model_path, device=device, num_gpus=args.device_num)

    if data_file.endswith("json"):
        data = load_json(data_file)
    else:
        data = load_jsonl(data_file)
    
    data_input = data[args.start_index:args.end_index] # we slice the original big data to different partitions.
    
    print("Number of samples:", len(data_input))
    model_name = args.model_name # the model name
    model_inference_config = model_info_dict[model_name]
    processed_data = process_data_with_chat_responses(data_input, model, tokenizer, device, model_inference_config, args)
    save_jsonl(output_file, processed_data)

    # save to excel
    save_cols = ["conversations", "meta", "response"]
    df = pd.DataFrame(processed_data)
    for col in save_cols:
        if col not in df.columns:
            df[col] = ""
    df = df[save_cols]
    df["response"] = df["response"].apply(lambda x: x.replace("</s>", ""))
    # df["response_last_10_words"] = df["response"].apply(lambda x: x[-20:])
    # df["analysis_last_10_words"] = df["analysis"].apply(lambda x: x[-20:])
    df.to_excel(output_file.replace(".json", ".xlsx"), index=False, engine="xlsxwriter")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="chatglm2-6b", help="Name of the model"
    )
    parser.add_argument("--model_path", type=str, default="/mnt/pfs/zitao_team/tianqiaoliu/public_github/ChatGLM2-6B/ptuning/output/mathgpt-chatglm2-6b-ft-2e-5/checkpoint-POINTNUM")
    parser.add_argument("--data_file", type=str, help="Path to the input data file", default="/mnt/pfs/zitao_team/big_model/processed_data/test_data_junior_small.json")
    parser.add_argument("--output_file", type=str, help="Path to the output file",default='./results/chatglm2-6b/test_data_small_with_response_chatglm2_POINTNUM.json')
    parser.add_argument("--start_index", type=int, help="Where to start the slice of the dataset")
    parser.add_argument("--end_index", type=int, help="The size of the slice of the dataset")
    parser.add_argument("--gpu_id", type=str, default="7", help="ID of the GPU to use")
    parser.add_argument("--device_num", type=int, default=1, help="number of gpus to use")
    parser.add_argument("--max_new_tokens", type=int, help="The maximum num of max new tokens")
    parser.add_argument("--stop_str", type=str, default="", help="the stop string for the model for this dataset")
    parser.add_argument('--few_shot', action='store_true', help='whether to activate few shot or not')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    generate_chat_responses(
        args.model_path,
        args.data_file,
        args.output_file.format(model_name=args.model_name),
        args,
    )