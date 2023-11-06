import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from collections import defaultdict


class DatasetBuilder:
    def __init__(self, base_dir, output_dir):
        self.base_dir = base_dir
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    @staticmethod
    def safe_format(s, **kwargs):
        while True:
            try:
                s = s.format(**kwargs)
                break
            except KeyError as e:
                missing_key = str(e).strip("'")
                kwargs[missing_key] = ""
        return s

    @staticmethod
    def build_one_turn_conversation(prompt, line, roles=("human", "gpt")):
        return [{"from": role, "value": prompt[role].format(**line)} for role in roles]

    def build_n_shot_conversation(
        self, prompt, line, N_shot_lines, roles=("human", "gpt")
    ):
        conversations = []
        for shot in N_shot_lines:
            conversations.extend(self.build_one_turn_conversation(prompt, shot, roles))
        conversations.extend(
            self.build_one_turn_conversation(prompt, line, roles=("human",))
        )
        # conversations.append(
        #     {
        #         "from": "gpt",
        #         "value": self.safe_format(prompt["gpt"], **defaultdict(str)),
        #     }
        # )
        return conversations

    def build_n_shot_dataset(self, prompt, dataset, N):
        dataset_list = []
        for idx, line in enumerate(dataset):
            n_shot_lines = dataset[-N:] if idx < N else dataset[:N]
            dataset_list.append(
                {
                    "conversations": self.build_n_shot_conversation(
                        prompt, line, n_shot_lines
                    ),
                    "meta": line.copy(),
                }
            )
        return dataset_list

    def build_datasets(self, role_prompt, dataset, N, name):
        normalized_dataset = self.build_n_shot_dataset(role_prompt, dataset, N)
        output_path = os.path.join(self.output_dir, f"{name}-{N}shot.jsonl")
        pd.DataFrame(normalized_dataset).to_json(
            output_path, orient="records", force_ascii=False, lines=True
        )
        print(f"{output_path} built. Total {len(normalized_dataset)} samples.")

    def process_dataset(self, dataset_config):
        dataset = dataset_config["loading_func"](dataset_config["path"])
        for N in dataset_config["shots"]:
            self.build_datasets(
                dataset_config["role_prompt"], dataset, N, dataset_config["name"]
            )

    def process(self, config):
        for dataset_config in config:
            self.process_dataset(dataset_config)


def read_one_csv(file):
    return pd.read_csv(file, encoding="utf-8").to_dict("records")


def read_one_json(file):
    return json.load(open(file, "r", encoding="utf-8"))


def read_one_jsonlines(file):
    return [json.loads(line) for line in open(file, "r", encoding="utf-8")]


def read_special_csv(args):
    name, file = args
    df = pd.read_csv(file, encoding="utf-8")
    df["subset"] = name
    return df.to_dict("records")


def gsm8k_loading_func(file):
    dataset = [json.loads(line) for line in open(file, "r", encoding="utf-8")]
    for d in dataset:
        d["direct_answer"] = d["answer"].split("#### ")[1].replace(",", "")
        d["answer_prefix"] = d["answer"].split("#### ")[0].strip("\n")
    return dataset


def mmlu_loading_func(args):
    datasets = []
    for name, file in args:
        dataset = pd.read_csv(
            file,
            encoding="utf-8",
            header=None,
            names=["input", "A", "B", "C", "D", "target"],
        )
        dataset["subset"] = name
        datasets.extend(dataset.to_dict("records"))
    return datasets


def bba_loading_func(args):
    datasets = []
    for name, file in args:
        dataset = json.load(open(file, "r", encoding="utf-8"))["examples"]
        for d in dataset:
            d["subset"] = name
        datasets.extend(dataset)
    return datasets


def bbh_loading_func(file):
    return json.load(open(file, "r", encoding="utf-8"))["examples"]


def scq_loading_func(file):
    dataset = []
    data = [json.loads(line) for line in open(file, "r", encoding="utf-8")]

    def format_line(line):
        line = line.copy()
        new_line = {}
        new_line["id"] = line["queId"]
        answer_option_list = [
            item[0]["aoVal"] + ": " + item[0]["content"]
            for item in line["answer_option_list"]
        ]
        new_line["problem"] = (
            line["problem"].strip() + " " + ";".join(answer_option_list)
        )
        new_line["answer"] = line["answer_value"]
        new_line['options'] = '\n'.join(answer_option_list)
        return new_line

    for row in data:
        line = format_line(row)
        dataset.append(line)
    return dataset


def math_loading_func(file):
    math = json.load(open(file, "r"))
    lines = []
    for k, v in math.items():
        v.update({"path": k})
        lines.append(v)
    return lines


def arith_std_loading_func(args):
    datasets = []
    for name, file in args:
        dataset = json.load(open(file, "r", encoding="utf-8"))
        for d in dataset:
            d["subset"] = name
        datasets.extend(dataset)
    return datasets

def draw_loading_func(file):
    dataset = json.load(open(file, "r", encoding="utf-8"))
    for d in dataset:
        # list to str, remove the brackets
        d["answer"] = str(d["ans"]).replace("[", "").replace("]", "")
    return dataset    
    
def hmwp_loading_func(file):
    dataset = json.load(open(file, "r", encoding="utf-8"))
    for d in dataset:
        d["original_text"] = d["original_text"].replace(" ", "")
        d["ans"] = ','.join(map(str,d['ans']))
    return dataset


def gaokao_loading_func(args):
    datasets = []
    for name, file in args:
        dataset = json.load(open(file, "r", encoding="utf-8"))["example"]
        for d in dataset:
            d["subset"] = name
            d['answer'] = ','.join(d['answer'])
        datasets.extend(dataset)
    return datasets


def agieval_loading_func(args):
    name, file = args
    datasets = []
    data = [json.loads(line) for line in open(file, "r", encoding="utf-8")]
    for row in data:
        ### 以下是AGIEval的数据处理
        passage = row["passage"] if row["passage"] else ""
        question = passage + row["question"]
        options = "\n".join(row["options"]) if row["options"] else ""
        if row["label"]:
            if isinstance(row["label"], list):
                label = "".join(row["label"])
            else:
                label = row["label"]
        else:
            label = row["answer"]
        datasets.append(
            {
                "question": question,
                "options": options,
                "label": label,
                "subset": name,
            }
        )
        ### 以上是AGIEval的数据处理
    return datasets

# 选择题 cmmlu AGIEval ceval scq_ch scq_en GAOKAO-BENCH mmlu MathQA
configs = [
    {
        # math23k
        # https://arxiv.org/pdf/2109.03034v1.pdf
        "name": "math23k",
        "path": "math23k/testset.json",
        "role_prompt": {"human": "问题:{original_text}\n答案:", "gpt": "{ans}"},
        "shots": [0, 3],
        "loading_func": read_one_json,
        "answer_column": "ans",
    },
    {
        # MathQA
        # https://arxiv.org/pdf/1907.01642v1.pdf
        "name": "MathQA",
        "path": "MathQA/test.json",
        "role_prompt": {
            "human": "There is a multiple choice question:\nQuestion:{Problem}\n{options}\nPlease give your answer from the five options a, b, c, d, e.\nAnswer:",
            "gpt": "{correct}",
        },
        "shots": [0, 3],
        "loading_func": read_one_json,
        "answer_column": "correct",
    },
    {
        # ape210k
        # https://github.com/Chenny0808/ape210k
        "name": "ape210k",
        "path": "ape210k/test.ape.json",
        "role_prompt": {
            "human": "下面是一个数学题，根据问题回答。问题:{original_text}答案:",
            "gpt": "{ans}",
        },
        "shots": [0, 3],
        "loading_func": read_one_jsonlines,
        "answer_column": "ans",
    },
    {
        # GSM8K
        # https://arxiv.org/pdf/2110.14168.pdf
        "name": "GSM8K",
        "path": "GSM8K/test.jsonl",
        "role_prompt": {
            "human": "Calculate the following math word problem, give your step-by-step solution first and then show your final answer: {question}\n",
            "gpt": "{answer_prefix}\nAnswer: {direct_answer}",
        },
        "shots": [0, 8],
        "loading_func": gsm8k_loading_func,
        "answer_column": "direct_answer",
    },
    {
        # mmlu
        # https://arxiv.org/pdf/2009.03300.pdf
        "name": "mmlu",
        "path": tuple(
            (name, f"mmlu/test/{name}_test.csv")
            for name in (
                "abstract_algebra",
                "college_mathematics",
                "elementary_mathematics",
                "high_school_mathematics",
            )
        ),
        "role_prompt": {
            "human": "There is a single choice question:\nQuestion: {input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nPlease give your answer from the four options A, B, C, D.\nAnswer:",
            "gpt": "{target}",
        },
        "shots": [0, 3],
        "loading_func": mmlu_loading_func,
        "answer_column": "target",
    },
    {
        # bb_arithmetics
        # https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/arithmetic
        "name": "bb_arithmetics",
        "path": tuple(
            (name, f"bb_arithmetic/{name}/task.json")
            for n in range(1, 6)
            for name in (
                f"{n}_digit_addition",
                f"{n}_digit_division",
                f"{n}_digit_multiplication",
                f"{n}_digit_subtraction",
            )
        ),
        "role_prompt": {
            "human": "Here is a math question, you need answer the question. Question:{input}\nThe correct answer is ",
            "gpt": "{target}",
        },
        "shots": [0, 3],
        "loading_func": bba_loading_func,
        "answer_column": "target",
    },
    {
        # arith_std
        # from TAL
        "name": "arith_std",
        "path": tuple(
            (name, f"arith_std/{name}.json")
            for name in (
                "sin_standard",
                "cos_standard",
                "tan_standard",
                "log_standard",
                "sqrt_standard",
                "pow_standard",
                "decimal_standard",
                "mod_fact_standard",
                "low_digits_standard",
                "high_digits_standard",
                "prior_arith_standard",
                "compound_op_standard",
                "mixed_advanced_standard",
                "mixed_compound_standard",
                "special_cases_standard",
            )
        ),
        "role_prompt": {
            "human": "Calculate the following arithmetic problem: {expression}\nAnswer: ",
            "gpt": "{answer}",
        },
        "shots": [0, 3],
        "loading_func": arith_std_loading_func,
        "answer_column": "answer",
    },
    {
        # GAOKAO-BENCH
        # https://arxiv.org/pdf/2305.12474v2.pdf
        "name": "GAOKAO-BENCH",
        "path": tuple(
            (name, f"GAOKAO-BENCH/data/Multiple-choice_Questions/{name}.json")
            for name in (
                "2010-2022_Math_I_MCQs",
                "2010-2022_Math_II_MCQs",
            )
        ),
        "role_prompt": {
            "human": "请你做一道数学选择题\n题目如下：{question}\n请从A,B,C,D四个选项中选出正确答案。\n答案是: ",
            "gpt": "{answer}",
        },
        "shots": [0, 3],
        "loading_func": gaokao_loading_func,
        "answer_column": "answer",
    },
    {
        # mawps
        # https://aclanthology.org/N16-1136.pdf
        "name": "mawps",
        "path": "mawps/testset.json",
        "role_prompt": {
            "human": "Calculate the following math word problem: {original_text}\nAnswer:",
            "gpt": "{ans}",
        },
        "shots": [0, 3],
        "loading_func": read_one_json,
        "answer_column": "ans",
    },
    {
        # BBH
        # https://arxiv.org/pdf/2210.09261.pdf
        "name": "BBH",
        "path": "BBH/data/multistep_arithmetic_two.json",
        "role_prompt": {
            "human": "Please answer the following question.\nQ: {input}\nA:",
            "gpt": "{target}",
        },
        "shots": [0, 3],
        "loading_func": bbh_loading_func,
        "answer_column": "target",
    },
    {
        # scq_en
        # https://huggingface.co/datasets/math-eval/TAL-SCQ5K
        "name": "scq_en",
        "path": "scq_en/en_single_choice_test.jsonl",
        "role_prompt": {
            "human": "Solve the following math word problem and choose a final choice among the provided choices A,B,C,D,E : {problem}\nAnswer:",
            "gpt": "{answer}",
        },
        "shots": [0, 3],
        "loading_func": scq_loading_func,
        "answer_column": "answer",
    },
    {
        # scq_ch
        # https://huggingface.co/datasets/math-eval/TAL-SCQ5K
        "name": "scq_ch",
        "path": "scq_ch/ch_single_choice_test.jsonl",
        "role_prompt": {
            "human": "计算以下数学单选题并给出最终选项 A,B,C,D,E 之一: {problem}\n答案:",
            "gpt": "{answer}",
        },
        "shots": [0, 3],
        "loading_func": scq_loading_func,
        "answer_column": "answer",
    },
    {
        # math
        # https://arxiv.org/pdf/2103.03874.pdf
        "name": "math",
        "path": "math/math.json",
        "role_prompt": {
            "human": "Solve the problem and give your step-by-step solution:\n{problem}\nSolution:",
            "gpt": "{solution}",
        },
        "shots": [0, 4],
        "loading_func": math_loading_func,
        "answer_column": "solution",
    },
    {
        # asdiv-a
        # https://github.com/LYH-YF/MWPToolkit/blob/master/dataset/asdiv-a/
        "name": "asdiv-a",
        "path": "asdiv-a/asdiv-a.jsonl",
        "role_prompt": {
            "human": "Calculate the following math word problem: {problem}\nAnswer:",
            "gpt": "{answer}",
        },
        "shots": [0, 3],
        "loading_func": read_one_jsonlines,
        "answer_column": "answer",
    },
    {
        # svamp
        # https://github.com/arkilpatel/SVAMP
        "name": "svamp",
        "path": "svamp/svamp_test.jsonl",
        "role_prompt": {
            "human": "Calculate the following math word problem: {problem}\nAnswer:",
            "gpt": "{answer}",
        },
        "shots": [0, 3],
        "loading_func": read_one_jsonlines,
        "answer_column": "answer",
    },
    {
        # math401
        # https://github.com/GanjinZero/math401-llm
        "name": "math401",
        "path": "math401-llm/math401.json",
        "role_prompt": {
            "human": "计算下面的数学题: {query} \n答案:",
            "gpt": "{response}",
        },
        "shots": [0, 3],
        "loading_func": read_one_json,
        "answer_column": "response",
    },
    {
        # draw
        # https://github.com/LYH-YF/MWPToolkit/blob/master/dataset/draw/
        "name": "draw",
        "path": "draw/testset.json",
        "role_prompt": {
            "human": "Calculate the following math problem: {original_text}\nAnswer:",
            "gpt": "{answer}",
        },
        "shots": [0, 3],
        "loading_func": draw_loading_func,
        "answer_column": "answer",
    },
    {
        # dolphin1878
        # https://www.microsoft.com/en-us/research/project/sigmadolphin/
        "name": "dolphin1878",
        "path": "dolphin1878/testset.json",
        "role_prompt": {
            "human": "Calculate the following math problem: {text}\nAnswer:",
            "gpt": "{ans}",
        },
        "shots": [0, 3],
        "loading_func": read_one_json,
        "answer_column": "ans",
    },
    {
        # hmwp
        # https://github.com/QinJinghui/SAU-Solver
        "name": "hmwp",
        "path": "hmwp/testset.json",
        "role_prompt": {
            "human": "回答下面的数学题:\n{original_text}\n答案:",
            "gpt": "{ans}",
        },
        "shots": [0, 3],
        "loading_func": hmwp_loading_func,
        "answer_column": "ans",
    },
]

special_configs = [
    {
        # ceval
        # https://github.com/SJTU-LIT/ceval
        "name": "ceval",
        "prompt_dict": {
            "advanced_mathematics": {
                "human": "以下是中国关于高等数学考试的单项选择题。\n{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n请从A,B,C,D四个选项中选出正确答案。\n答案是: ",
                "gpt": "{answer}",
            },
            "discrete_mathematics": {
                "human": "以下是中国关于离散数学考试的单项选择题。\n{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n请从A,B,C,D四个选项中选出正确答案。\n答案是: ",
                "gpt": "{answer}",
            },
            "high_school_mathematics": {
                "human": "以下是中国关于高中数学考试的单项选择题。\n{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n请从A,B,C,D四个选项中选出正确答案。\n答案是: ",
                "gpt": "{answer}",
            },
            "middle_school_mathematics": {
                "human": "以下是中国关于初中数学考试的单项选择题。\n{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n请从A,B,C,D四个选项中选出正确答案。\n答案是: ",
                "gpt": "{answer}",
            },
            "probability_and_statistics": {
                "human": "以下是中国关于概率统计考试的单项选择题。\n{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n请从A,B,C,D四个选项中选出正确答案。\n答案是: ",
                "gpt": "{answer}",
            },
        },
        "path": tuple(
            (key, f"ceval/formal_ceval/test/{key}_test.csv")
            for key in (
                "advanced_mathematics",
                "discrete_mathematics",
                "high_school_mathematics",
                "middle_school_mathematics",
                "probability_and_statistics",
            )
        ),
        "shots": [0],
        "loading_func": read_special_csv,
        "answer_column": "answer",
    },
    {
        # AGIEval
        # https://github.com/microsoft/AGIEval/blob/main/
        "name": "AGIEval",
        "prompt_dict": {
            "gaokao-mathqa": {
                "human": "以下是一道中国高考数学选择题。\n{question}\n{options}\n请从A,B,C,D四个选项中选出正确答案。\n答案是：",
                "gpt": "{label}",
            },
            "sat-math": {
                "human": "The following is a SAT Math question. \n{question}\n{options}\nPlease give your answer from the four options A, B, C, D.\nThe answer is",
                "gpt": "{label}",
            },
            "aqua-rat": {
                "human": "The following is a AQUA-RAT question. \n{question}\n{options}\nPlease give your answer from the four options A, B, C, D, E.\nThe answer is",
                "gpt": "{label}",
            },
            "gaokao-mathcloze": {
                "human": "以下是一道中国高考数学填空题，请填入正确的答案。\n{question}\n答案是：",
                "gpt": "{label}",
            },
            "math": {
                "human": "The following is a Math question. Please answer the question.\n{question}\nThe answer is",
                "gpt": "{label}",
            },
        },
        "path": tuple(
            (key, f"AGIEval/data/v1/{key}.jsonl")
            for key in (
                "gaokao-mathqa",
                "sat-math",
                "aqua-rat",
                "gaokao-mathcloze",
                "math",
            )
        ),
        "shots": [0, 3],
        "loading_func": agieval_loading_func,
        "answer_column": "label",
    },
    {
        # cmmlu
        # https://github.com/haonan-li/CMMLU/blob/master
        "name": "cmmlu",
        "prompt_dict": {
            "college_mathematics": {
                "human": "以下是关于大学数学的单项选择题。\n题目：{Question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n请从A,B,C,D四个选项中选出正确答案。\n答案是：",
                "gpt": "{Answer}",
            },
            "elementary_mathematics": {
                "human": "以下是关于初等数学的单项选择题。\n题目：{Question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n请从A,B,C,D四个选项中选出正确答案。\n答案是：",
                "gpt": "{Answer}",
            },
            "high_school_mathematics": {
                "human": "以下是关于高中数学的单项选择题。\n题目：{Question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n请从A,B,C,D四个选项中选出正确答案。\n答案是：",
                "gpt": "{Answer}",
            },
        },
        "path": tuple(
            (key, f"cmmlu/test/{key}.csv")
            for key in (
                "college_mathematics",
                "elementary_mathematics",
                "high_school_mathematics",
            )
        ),
        "shots": [0, 3],
        "loading_func": read_special_csv,
        "answer_column": "Answer",
    },
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./datasets/")
    parser.add_argument("--output_dir", type=str, default="./")
    args = parser.parse_args()

    # modify the path
    for config in configs:
        if isinstance(config["path"], str):
            config["path"] = os.path.join(args.input_dir, config["path"])
        elif isinstance(config["path"], tuple):
            config["path"] = (
                (name, os.path.join(args.input_dir, file))
                for name, file in config["path"]
            )
        else:
            raise ValueError("Invalid path type.")

    for config in special_configs:
        config["path"] = (
            (name, os.path.join(args.input_dir, file)) for name, file in config["path"]
        )

    # Process the datasets
    builder = DatasetBuilder(args.input_dir, args.output_dir)
    builder.process(configs)

    # Process the special datasets
    for config in special_configs:
        paths = list(config["path"]).copy()
        for N in config["shots"]:
            normalized_dataset = []
            for name, file in paths:
                datasets = config["loading_func"]((name, file))
                normalized_dataset.extend(builder.build_n_shot_dataset(config["prompt_dict"][name], datasets, N))
            output_path = os.path.join(args.output_dir, f"{config['name']}-{N}shot.jsonl")
            pd.DataFrame(normalized_dataset).to_json(output_path, orient="records", force_ascii=False, lines=True)
            print(f"{output_path} built. Total {len(normalized_dataset)} samples.")
