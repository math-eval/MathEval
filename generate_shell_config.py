#!/usr/bin/env python

import os

# change the data_dir with your own path
data_dir = "data"

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
        "data_path": os.path.join(data_dir, "BBH-0shot.jsonl"),
        "data_name": "BBH-0shot",
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
        "model_path": "/mnt/pfs/zitao_team/big_model/raw_models/Qwen-14B-Chat",
        "model_name": "Qwen-14B-Chat",
        "num_gpu": 1,
        "template_name": "qwen-7b-chat",
    }
]
