import os
import argparse
import importlib.util
import subprocess
from tqdm import tqdm
from generate_inference_shell_scripts.generate_shell_models import generate_shell_scripts
from generate_inference_shell_scripts.generate_shell_models_vllm import generate_shell_scripts_vllm

def load_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main(args):
    config_module = load_module_from_file("generate_shell_config", args.configs)
    all_data_info = config_module.all_data_info
    root_model_info = config_module.root_model_info
    
    if args.accelerator == 'vllm':
        script_files = generate_shell_scripts_vllm(all_data_info, root_model_info)
    else:
        script_files = generate_shell_scripts(all_data_info, root_model_info)
    
    for script_file in tqdm(script_files, desc="Executing scripts"):
        print(f"Executing {script_file}")
        result = subprocess.run(["bash", script_file], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error executing {script_file}: {result.stderr}")
        else:
            print(f"Output of {script_file}: {result.stdout}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs", type=str, default="/mnt/pfs/zitao_team/fangzhensheng/MathEval/generate_shell_config.py", help="configs of matheval"
    )
    parser.add_argument(
        "--accelerator", 
        type=str, 
        default="", 
        help="Specify the accelerator for inference. Supported options: 'vllm'. Leave empty for default settings."
    )
    
    args = parser.parse_args()
    main(args)