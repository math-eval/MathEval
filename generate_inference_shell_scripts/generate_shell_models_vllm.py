#!/usr/bin/env python

import os
from datetime import datetime

def generate_shell_scripts_vllm(all_data_info, root_model_info):
    date = datetime.now().strftime("%Y_%m%d")
    output_dir = "./result_cot" # the generated result path
    where_to_save_shell = "./generated_shell_cot" # the path to save model inference shell scripts
    log_file = "./logs_cot" # the dir to save log files.
    
    script_file_names = []

    for one_model_info in root_model_info:
        model_name = one_model_info["model_name"]
        model_path = one_model_info["model_path"]
        device_num = one_model_info["num_gpu"]

        log_dir_model_data = os.path.join(log_file, date, model_name)
        os.makedirs(log_dir_model_data, exist_ok=True)

        output_dir_model_data = os.path.join(output_dir, date, model_name)
        os.makedirs(output_dir_model_data, exist_ok=True)

        script_folder = os.path.join(where_to_save_shell, date, model_name)
        os.makedirs(script_folder, exist_ok=True)
        script_file_name = os.path.join(script_folder, model_name + ".sh")
        script_file_names.append(script_file_name)

        with open(script_file_name, "w") as script_file:
            script_file.write("#!/bin/bash\n\n")
            gpu_id_assignment = ",".join(map(str, range(device_num)))
            script_file.write(f'output_dir="{output_dir_model_data}"\n')
            script_file.write(f'gpu_ids="{gpu_id_assignment}"\n')
            script_file.write(f"model_path={model_path}\n")
            script_file.write(f"model_name={model_name}\n")
            script_file.write(f"device_num={device_num}\n")
            script_file.write(f"log_dir_model_data={log_dir_model_data}\n")

            script_file.write(f'mkdir -p "$output_dir"\n')
            script_file.write(
                f"pids=() # Array to store the PIDs of the background processes\n\n"
            )

            #script_file.write("for ((i=0; i<${#gpu_ids[@]}; i++)); do\n")
            #script_file.write("    gpu_id=${gpu_ids[$i]}\n")

            script_file.write(
                f'echo "开始评估模型: $model_name, GPU ID: $gpu_ids, Start index: $start_index, End index: $end_index"\n'
            )

            script_file.write(f'output_dir="$output_dir"\n')

            script_file.write(f'log_file="$log_dir_model_data/${{model_name}}.log"\n\n')

            script_file.write(
                'command="CUDA_VISIBLE_DEVICES=$gpu_ids nohup python3 -u ./inference_matheval.py'
            )  # 改inference的python文件
            script_file.write(
                f" --model_name $model_name --model_path $model_path --device_num $device_num"
            )
            script_file.write(' --accelerator vllm')
            script_file.write(
                f' --output_dir $output_dir --max_new_tokens 512 > $log_file 2>&1 &"\n\n'
            )
            
            script_file.write('eval "$command"\n')
            script_file.write(
                "pids+=($!) # Store the PID of the last command run in the background\n\n"
            )
            script_file.write('for pid in "${pids[@]}"; do\n')
            script_file.write('    wait "$pid"\n')
            script_file.write("done\n\n")

            script_file.write(
                "# Kill all processes related to evaluating the model no api\n"
            )
            script_file.write("pkill -f inference_matheval.py\n\n")

            script_file.write(
                'echo "All evaluations completed and related processes killed."\n'
            )

        # Make the script executable
        os.chmod(script_file_name, 0o755)
    
    return script_file_names

# Call the function and print the returned script file names
if __name__ == "__main__":
    script_files = generate_shell_scripts()
    for script_file in script_files:
        print(script_file)