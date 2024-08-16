import os
import argparse
import importlib.util
import subprocess
from tqdm import tqdm
import glob

from transformers import AutoModelForCausalLM, AutoTokenizer

chat_prompt = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>human
{}<|im_end|>
<|im_start|>gpt
"""

basic_prompt = """## 任务描述\n\n你是一个数学老师，学生提交了题目的解题步骤，你需要参考`题干`，`解析`和`答案`，判断`学生解题步骤`的结果是否正确。忽略`学生解题步骤`中的错误，只关注最后的答案。答案可能出现在`解析`中，也可能出现在`答案`中。\n\n## 输入内容\n\n题干:\n\n```\n{{question}}\n```\n\n解析:\n\n```\n{{analysis}}\n\n```\n\n答案:\n\n```\n{{answer}}\n```\n\n学生解题步骤:\n\n```\n{{pred_step}}\n```\n\n输出:"""
base_prompt = chat_prompt.format(basic_prompt)

def build_user_query(question, pred_answer, answer, base_prompt):
    input_text = base_prompt.replace("{{question}}", question)
    input_text = input_text.replace("{{pred_step}}", pred_answer)
    input_text = input_text.replace("{{answer}}", answer)
    input_text = input_text.replace("{{analysis}}", "") # default set analysis to blank, if exist, you can pass in the corresponding parameter.
    return input_text


def process_data_with_chat_responses(data, model, tokenizer, device, model_inference_config, args):
    processed_data = []
    inference_mode = "few-shot" if args.few_shot else "zero-shot"
    print(f"[INFO]: Activate {inference_mode} inference.")
    
    template_name = model_inference_config['template_name']
    
    if args.accelerator == 'vllm':
        prompts_list = []
        for item in tqdm(data):
            prompt = build_user_query(item["conversations"], template_name, args)
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
    
    print("Number of samples:", len(data_input))
    model_name = args.model_name # the model name
    model_inference_config = model_info_dict[model_name]
    processed_data = process_data_with_chat_responses(data_input, model, tokenizer, device, model_inference_config, args)
    save_jsonl(output_file, processed_data)
    
    
def generate_chat_responses_all(args):
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    print("Device to inference: {}".format(device))
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
        output_file = os.path.join(args.output_dir, base_data_name + ".json")
        generate_chat_responses(model, tokenizer, one_data_info, output_file, device, args)
        

def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created directory: {args.output_dir}")
    else:
        print(f"Directory already exists: {args.output_dir}")
    all_data_info = glob.glob(os.path.join(data_path, '**', '*.json'), recursive=True)
    generate_chat_responses_all(args):
    
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--input_dir", type=str, default="/mnt/pfs/zitao_team/fangzhensheng/MathEval/result_cot/Qwen-14B-Chat")
    parser.add_argument("--output_dir", type=str, help="Path to the output file", default="/mnt/pfs/zitao_team/fangzhensheng/MathEval/compare_result/Qwen-14B-Chat")
    parser.add_argument(
        "--accelerator", 
        type=str, 
        default="", 
        help="Specify the accelerator for inference. Supported options: 'vllm'. Leave empty for default settings."
    )
    
    args = parser.parse_args()
    main(args)