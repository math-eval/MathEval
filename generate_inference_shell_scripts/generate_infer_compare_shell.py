import os
from datetime import datetime
output_dir_infer = "/mnt/pfs/zitao_team/fangzhensheng/MathEval/all_infer_result"
output_dir_compare = "/mnt/pfs/zitao_team/fangzhensheng/MathEval/all_compare_result"
log_dir="/mnt/pfs/zitao_team/fangzhensheng/MathEval/all_infer_compare_logs"
where_to_save_shell = "/mnt/pfs/zitao_team/fangzhensheng/MathEval/all_infer_compare_shell"

def generate_infer_compare_shell(all_data_info, root_model_info, args):
    accelerator = args.accelerator
    mode = args.mode
    date = datetime.now().strftime("%Y_%m%d")
    script_file_names = []
    
    # Step 1: Generate and execute inference and comparison scripts
    for one_model_info in root_model_info:
        model_name = one_model_info["model_name"]
        model_path = one_model_info["model_path"]
        device_num = one_model_info["num_gpu"]

        log_dir_model_data = os.path.join(log_dir, date, model_name)
        os.makedirs(log_dir_model_data, exist_ok=True)

        output_dir_model_data_infer = os.path.join(output_dir_infer, date, model_name)
        os.makedirs(output_dir_model_data_infer, exist_ok=True)

        output_dir_model_data_compare = os.path.join(output_dir_compare, date, model_name)
        os.makedirs(output_dir_model_data_compare, exist_ok=True)

        script_folder = os.path.join(where_to_save_shell, date, model_name)
        os.makedirs(script_folder, exist_ok=True)
        script_file_name = os.path.join(script_folder, model_name + ".sh")
        script_file_names.append(script_file_name)

        with open(script_file_name, "w") as script_file:
            script_file.write("#!/bin/bash\n\n")
            
            gpu_id_assignment = ",".join(map(str, range(device_num)))

            # Inference variables
            script_file.write(f'output_dir_infer="{output_dir_model_data_infer}"\n')
            script_file.write(f'gpu_ids="{gpu_id_assignment}"\n')
            script_file.write(f"model_path={model_path}\n")
            script_file.write(f"model_name={model_name}\n")
            script_file.write(f"device_num={device_num}\n")
            script_file.write(f"accelerator={accelerator}\n")
            log_file = os.path.join(log_dir_model_data, model_name + ".log")
            script_file.write(f"log_file={log_file}\n")
            
            script_file.write(f'mkdir -p "$output_dir_infer"\n')
            script_file.write(f"pids=() # Array to store the PIDs of the background processes\n\n")

            script_file.write(f'echo "开始评估模型: $model_name, GPU ID: $gpu_ids"\n')

            script_file.write('command_infer="CUDA_VISIBLE_DEVICES=$gpu_ids nohup python3 -u /mnt/pfs/zitao_team/fangzhensheng/MathEval/inference_matheval.py')
            script_file.write(f" --model_name $model_name --model_path $model_path --device_num $device_num")
            script_file.write(' --accelerator $accelerator')
            script_file.write(f' --output_dir $output_dir_infer --max_new_tokens 512 > $log_file 2>&1 &"\n\n')

            script_file.write('eval "$command_infer"\n')
            script_file.write("pids+=($!) # Store the PID of the last command run in the background\n\n")
            script_file.write('for pid in "${pids[@]}"; do\n')
            script_file.write('    wait "$pid"\n')
            script_file.write("done\n\n")

            script_file.write("# Kill all processes related to evaluating the model no api\n")
            script_file.write("pkill -f inference_matheval.py\n\n")

            script_file.write('echo "Inference completed and related processes killed."\n\n')
            
            if mode == 'eval':
                # Comparison variables
                input_dir_model = os.path.join(output_dir_infer, date, model_name)

                script_file.write(f'output_dir_compare="{output_dir_model_data_compare}"\n')
                script_file.write(f'input_dir="{input_dir_model}"\n')

                script_file.write(f'mkdir -p "$output_dir_compare"\n')
                script_file.write(f'pids=() # Array to store the PIDs of the background processes\n\n')
                script_file.write('command_compare="CUDA_VISIBLE_DEVICES=$gpu_ids nohup python3 -u /mnt/pfs/zitao_team/fangzhensheng/MathEval/compare_with_local_model.py')
                script_file.write(f' --device_num $device_num')
                script_file.write(f' --input_dir $input_dir --output_dir $output_dir_compare >> $log_file 2>&1 &"\n\n')

                script_file.write('eval "$command_compare"\n')
                script_file.write('pids+=($!) # Store the PID of the last command run in the background\n\n')
                script_file.write('for pid in "${pids[@]}"; do\n')
                script_file.write('    wait "$pid"\n')
                script_file.write("done\n\n")

                script_file.write("# Kill all processes related to evaluating the model no api\n")
                script_file.write("pkill -f compare_with_local_model.py\n\n")

                script_file.write('echo "Comparison completed and related processes killed."\n')

        # Make the script executable
        os.chmod(script_file_name, 0o755)
        
    return script_file_names