# json_utils.py
import json
import jsonlines

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