import subprocess
import time
import os
import glob
import threading
import xml.etree.ElementTree as ET
from collections import namedtuple, deque
import argparse

COMMAND_TEMPLATE = """
CUDA_VISIBLE_DEVICES={gpu_ids} python auto_eval_logits.py \
    --model_name "{model_model}" \
    --model_path "{model_path}" \
    --tokenizer_path "{tokenizer_path}" \
    --data_path "{data_path}" \
    --batch_size 4
"""

Task = namedtuple("Task", ["model_model", "model_path", "tokenizer_path", "data_path"])

parser = argparse.ArgumentParser(description="Script for text generation")
parser.add_argument('--model_path', required=True, help="Path to the model checkpoint")
parser.add_argument('--model_name', required=True, help="Name of the model checkpoint")
parser.add_argument('--tokenizer_path', required=True, help="Path to the tokenizer")
parser.add_argument('--data_dir', required=True, help="Dir to jsonl file")

args = parser.parse_args()

files = glob.glob(os.path.join(args.data_dir, '**/*.jsonl'), recursive=True)
# model_name = 'Mistral-7B-v0.1'
# model_path = '/mnt/pfs/zitao_team/big_model/raw_models/Mistral-7B-v0.1/'
# tokenizer_path = '/mnt/pfs/zitao_team/big_model/raw_models/Mistral-7B-v0.1/'
model_name = args.model_name
model_path = args.model_path
tokenizer_path = args.tokenizer_path

tasks = [
    Task(model_name, model_path, tokenizer_path, data_path) for data_path in files
]
REQUIRED_GPUS = 1
REQUIRED_MEMORY = 75000
LOCK_TIME = 300  # seconds to lock a GPU after launching a task

task_queue = deque(tasks)
print(task_queue)

def unlock_gpus_after_delay(gpu_ids, delay, locked_gpus_set):
    """Unlocks GPUs after a specified delay."""
    time.sleep(delay)
    for gpu_id in gpu_ids:
        locked_gpus_set.remove(gpu_id)
        
def get_free_gpus(required_memory=5000, locked_gpus=[]):
    result = subprocess.run(["nvidia-smi", "-q", "-x"], stdout=subprocess.PIPE)
    gpus = ET.fromstring(result.stdout).findall("gpu")

    free_gpus = []
    for gpu in gpus:
        memory = int(gpu.find("fb_memory_usage/free").text.split()[0])
        gpu_id = int(gpu.find("minor_number").text)
        
        if memory >= required_memory and gpu_id not in locked_gpus:
            free_gpus.append(gpu_id)

    return free_gpus


locked_gpus = set()  # a set to keep track of GPUs that are temporarily locked

threads = []  # 用于保存所有线程的列表
processes = []  # 用于保存所有子进程的列表
while task_queue:
    free_gpus = get_free_gpus(REQUIRED_MEMORY, locked_gpus=locked_gpus)

    if len(free_gpus) >= REQUIRED_GPUS:
        selected_gpus = free_gpus[:REQUIRED_GPUS]
        
        # Lock the GPUs
        for gpu in selected_gpus:
            locked_gpus.add(gpu)

        task = task_queue.popleft()
        print("*" * 80)
        print(f"Launching task {task} on GPUs {selected_gpus}")
        command = COMMAND_TEMPLATE.format(gpu_ids=",".join(map(str, selected_gpus)), **task._asdict())
        p = subprocess.Popen(command, shell=True)
        processes.append(p)  # 将新子进程添加到列表中
        
        # Create a new thread to unlock GPUs after LOCK_TIME
        t = threading.Thread(target=unlock_gpus_after_delay, args=(selected_gpus, LOCK_TIME, locked_gpus))
        t.start()
        threads.append(t)  # 将新线程添加到列表中
    else:
        print(f"Waiting for available GPUs. Current free GPUs: {len(free_gpus)}")
        time.sleep(60)  # check every 10 seconds

# 等待所有子线程完成
for t in threads:
    t.join()

# 等待所有子进程完成
for p in processes:
    p.wait()

print("All tasks completed!")
