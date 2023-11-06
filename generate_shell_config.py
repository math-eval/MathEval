#!/usr/bin/env python

import os

# change the data_dir with your own path
data_dir = ""

all_data_info = [
    {
        "data_path": os.path.join(data_dir, "scq_en-3shot.jsonl"),
        "data_name": "scq_en-3shot",
    },
    {
        "data_path": os.path.join(data_dir, "scq_ch-0shot.jsonl"),
        "data_name": "scq_ch-0shot",
    },
    {
        "data_path": os.path.join(data_dir, "ape210k-0shot.jsonl"),
        "data_name": "ape210k-0shot",
    },
    {
        "data_path": os.path.join(data_dir, "cmmlu-3shot.jsonl"),
        "data_name": "cmmlu-3shot",
    },
    {
        "data_path": os.path.join(data_dir, "asdiv-a-0shot.jsonl"),
        "data_name": "asdiv-a-0shot",
    },
    {
        "data_path": os.path.join(data_dir, "arith_std-0shot.jsonl"),
        "data_name": "arith_std-0shot",
    },
    {
        "data_path": os.path.join(data_dir, "AGIEval-0shot.jsonl"),
        "data_name": "AGIEval-0shot",
    },
    {
        "data_path": os.path.join(data_dir, "ape210k-3shot.jsonl"),
        "data_name": "ape210k-3shot",
    },
    {
        "data_path": os.path.join(data_dir, "mmlu-0shot.jsonl"),
        "data_name": "mmlu-0shot",
    },
    {
        "data_path": os.path.join(data_dir, "cmmlu-0shot.jsonl"),
        "data_name": "cmmlu-0shot",
    },
    {
        "data_path": os.path.join(data_dir, "GAOKAO-BENCH-3shot.jsonl"),
        "data_name": "GAOKAO-BENCH-3shot",
    },
    {
        "data_path": os.path.join(data_dir, "asdiv-a-3shot.jsonl"),
        "data_name": "asdiv-a-3shot",
    },
    {
        "data_path": os.path.join(data_dir, "scq_ch-3shot.jsonl"),
        "data_name": "scq_ch-3shot",
    },
    {
        "data_path": os.path.join(data_dir, "ceval-0shot.jsonl"),
        "data_name": "ceval-0shot",
    },
    {
        "data_path": os.path.join(data_dir, "GAOKAO-BENCH-0shot.jsonl"),
        "data_name": "GAOKAO-BENCH-0shot",
    },
    {
        "data_path": os.path.join(data_dir, "math23k-0shot.jsonl"),
        "data_name": "math23k-0shot",
    },
    {
        "data_path": os.path.join(data_dir, "MathQA-0shot.jsonl"),
        "data_name": "MathQA-0shot",
    },
    {
        "data_path": os.path.join(data_dir, "scq_en-0shot.jsonl"),
        "data_name": "scq_en-0shot",
    },
    {
        "data_path": os.path.join(data_dir, "mawps-0shot.jsonl"),
        "data_name": "mawps-0shot",
    },
    {
        "data_path": os.path.join(data_dir, "MathQA-3shot.jsonl"),
        "data_name": "MathQA-3shot",
    },
    {
        "data_path": os.path.join(data_dir, "dolphin1878-3shot.jsonl"),
        "data_name": "dolphin1878-3shot",
    },
    {
        "data_path": os.path.join(data_dir, "mawps-3shot.jsonl"),
        "data_name": "mawps-3shot",
    },
    {
        "data_path": os.path.join(data_dir, "bb_arithmetics-0shot.jsonl"),
        "data_name": "bb_arithmetics-0shot",
    },
    {
        "data_path": os.path.join(data_dir, "BBH-0shot.jsonl"),
        "data_name": "BBH-0shot",
    },
    {
        "data_path": os.path.join(data_dir, "arith_std-3shot.jsonl"),
        "data_name": "arith_std-3shot",
    },
    {
        "data_path": os.path.join(data_dir, "math401-0shot.jsonl"),
        "data_name": "math401-0shot",
    },
    {
        "data_path": os.path.join(data_dir, "svamp-0shot.jsonl"),
        "data_name": "svamp-0shot",
    },
    {
        "data_path": os.path.join(data_dir, "mmlu-3shot.jsonl"),
        "data_name": "mmlu-3shot",
    },
    {
        "data_path": os.path.join(data_dir, "hmwp-0shot.jsonl"),
        "data_name": "hmwp-0shot",
    },
    {
        "data_path": os.path.join(data_dir, "draw-0shot.jsonl"),
        "data_name": "draw-0shot",
    },
    {
        "data_path": os.path.join(data_dir, "draw-3shot.jsonl"),
        "data_name": "draw-3shot",
    },
    {
        "data_path": os.path.join(data_dir, "math23k-3shot.jsonl"),
        "data_name": "math23k-3shot",
    },
    {
        "data_path": os.path.join(data_dir, "dolphin1878-0shot.jsonl"),
        "data_name": "dolphin1878-0shot",
    },
    {
        "data_path": os.path.join(data_dir, "hmwp-3shot.jsonl"),
        "data_name": "hmwp-3shot",
    },
    {
        "data_path": os.path.join(data_dir, "math-0shot.jsonl"),
        "data_name": "math-0shot",
    },
    {
        "data_path": os.path.join(data_dir, "AGIEval-3shot.jsonl"),
        "data_name": "AGIEval-3shot",
    },
    {
        "data_path": os.path.join(data_dir, "math401-3shot.jsonl"),
        "data_name": "math401-3shot",
    },
    {
        "data_path": os.path.join(data_dir, "bb_arithmetics-3shot.jsonl"),
        "data_name": "bb_arithmetics-3shot",
    },
    {
        "data_path": os.path.join(data_dir, "GSM8K-0shot.jsonl"),
        "data_name": "GSM8K-0shot",
    },
    {
        "data_path": os.path.join(data_dir, "GSM8K-8shot.jsonl"),
        "data_name": "GSM8K-8shot",
    },
    {
        "data_path": os.path.join(data_dir, "BBH-3shot.jsonl"),
        "data_name": "BBH-3shot",
    },
    {
        "data_path": os.path.join(data_dir, "svamp-3shot.jsonl"),
        "data_name": "svamp-3shot",
    },
    {
        "data_path": os.path.join(data_dir, "math-4shot.jsonl"),
        "data_name": "math-4shot",
    },
]

root_model_info = [
    {
        "model_path": "/mnt/pfs/zitao_team/big_model/raw_models/Llama-2-7b-hf/",
        "model_name": "LLaMa2-7B",
        "num_gpu": 1,
        "template_name": "raw",  # name of the template in conversation
    },
    {
        "model_path": "/mnt/pfs/zitao_team/big_model/raw_models/Llama-2-7b-chat-hf/",
        "model_name": "LLaMa2-7B-chat",
        "num_gpu": 1,
        "template_name": "llama-2",  # name of the template in conversation
    },
    {
        "model_path": "/mnt/pfs/zitao_team/big_model/raw_models/Llama-2-13b-hf/",
        "model_name": "LLaMa2-13B",
        "num_gpu": 1,
        "template_name": "raw",  # name of the template in conversation
    },
    {
        "model_path": "/mnt/pfs/zitao_team/big_model/raw_models/Llama-2-13b-chat-hf/",
        "model_name": "LLaMa2-13B-chat",
        "num_gpu": 1,
        "template_name": "llama-2",
    },
    {
        "model_path": "/mnt/pfs/zitao_team/big_model/raw_models/Llama-2-70b-hf-0829/",
        "model_name": "LLaMa2-70B",
        "num_gpu": 2,
        "template_name": "raw",
    },
    {
        "model_path": "/mnt/pfs/zitao_team/big_model/raw_models/Llama-2-70b-chat-hf/",
        "model_name": "LLaMa2-70B-chat",
        "num_gpu": 2,
        "template_name": "llama-2",
    },
    {
        "model_path": "/mnt/pfs/zitao_team/big_model/raw_models/chatglm2-6b_2023.07.08/",
        "model_name": "chatglm2-6B",
        "num_gpu": 1,
        "template_name": "chatglm2",
    },
    {
        "model_path": "/mnt/pfs/zitao_team/big_model/raw_models/Baichuan2-13B-Base/",
        "model_name": "Baichuan2-13B",
        "num_gpu": 1,
        "template_name": "raw",
    },
    {
        "model_path": "/mnt/pfs/zitao_team/big_model/raw_models/Qwen-14B-Chat/",
        "model_name": "Qwen-14B-Chat",
        "num_gpu": 1,
        "template_name": "qwen-7b-chat",
    },
    {
        "model_path": "/mnt/pfs/zitao_team/big_model/raw_models/Qwen-14B/",
        "model_name": "Qwen-14B",
        "num_gpu": 1,
        "template_name": "raw",
    },
    {
        "model_path": "/mnt/pfs/zitao_team/big_model/raw_models/WizardMath-13B-V1.0/",
        "model_name": "WizardMath-13B",
        "num_gpu": 1,
        "template_name": "raw",
    },
    {
        "model_path": "/mnt/pfs/zitao_team/big_model/raw_models/WizardMath-70B-V1.0/",
        "model_name": "WizardMath-70B",
        "num_gpu": 2,
        "template_name": "raw",
    },
    {
        "model_path": "/mnt/pfs/zitao_team/big_model/raw_models/moss-moon-003-base/",
        "model_name": "moss-moon-003-base",
        "num_gpu": 1,
        "template_name": "raw",
    },
    {
        "model_path": "/mnt/pfs/zitao_team/big_model/raw_models/MAmmoTH-70B/",
        "model_name": "MAmmoTH-70B",
        "num_gpu": 2,
        "template_name": "raw",
    },
    {
        "model_path": "/mnt/pfs/zitao_team/big_model/raw_models/GAIRMath-Abel-70b/",
        "model_name": "GAIRMath-Abel-70b",
        "num_gpu": 2,
        "template_name": "raw",
    },
    {
        "model_path": "/mnt/pfs/zitao_team/big_model/raw_models/InternLM-20B/",
        "model_name": "InternLM-20B",
        "num_gpu": 1,
        "template_name": "raw",
    },
    {
        "model_path": "/mnt/pfs/zitao_team/big_model/raw_models/internlm-chat-20b/",
        "model_name": "internlm-chat-20b",
        "num_gpu": 1,
        "template_name": "internlm-chat",
    },
    {
        "model_path": "/mnt/lck_cfs/tianqiaoliu/models/llama2-70B-only-math-continue-continue-big-steps-1008/checkpoint-300/",
        "model_name": "LLama-2-Tianqiao",
        "num_gpu": 2,
        "template_name": "vicuna_v1.1",
    },
    {
        "model_path": "",
        "model_name": "GPT4",
        "num_gpu": 1,
        "template_name": "gpt4",
    },
    {
        "model_path": "",
        "model_name": "GPT35",
        "num_gpu": 1,
        "template_name": "gpt35",
    },
    {
        "model_path": "",
        "model_name": "spark",
        "num_gpu": 1,
        "template_name": "spark",
    },
    {
        "model_path": "",
        "model_name": "wenxin",
        "num_gpu": 1,
        "template_name": "wenxin",
    },
    {
        "model_path": "",
        "model_name": "mathgpt",
        "num_gpu": 1,
        "template_name": "mathgpt",
    },
        {
        "model_path": "/mnt/pfs/zitao_team/big_model/raw_models/MetaMath-70B-V1.0/",
        "model_name": "MetaMath-70B",
        "num_gpu": 2,
        "template_name": "raw",
    },
    {
        "model_path": "/mnt/pfs/zitao_team/big_model/raw_models/Mistral-7B-v0.1/",
        "model_name": "Mistral-7B",
        "num_gpu": 1,
        "template_name": "raw",
    },
    {
        "model_path": "/mnt/pfs/zitao_team/big_model/raw_models/Mistral-7B-Instruct-v0.1/",
        "model_name": "Mistral-7B-Instruct",
        "num_gpu": 1,
        "template_name": "mistral",
    },
    {
        "model_path": "/mnt/pfs/zitao_team/big_model/raw_models/llemma_7b/",
        "model_name": "llemma_7b",
        "num_gpu": 1,
        "template_name": "raw",
    },
    {
        "model_path": "/mnt/pfs/zitao_team/big_model/raw_models/llemma_34b/",
        "model_name": "llemma_34b",
        "num_gpu": 2,
        "template_name": "raw",
    }
]
