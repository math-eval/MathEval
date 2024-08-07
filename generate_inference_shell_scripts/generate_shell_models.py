#!/usr/bin/env python
import os

def generate_shell_scripts(all_data_info, root_model_info):
    output_dir = "./result_cot" # the generated result path
    where_to_save_shell = ("./generated_shell_cot") # the path to save model inference shell scripts
    log_file = "./logs_cot" # the dir to save log files.
    partition_size = 500 # Slice the dataset into different partitions

    script_file_names = []  # List to store script file names

    for one_data_info in all_data_info:
        for one_model_info in root_model_info:
            model_name = one_model_info["model_name"]
            model_path = one_model_info["model_path"]
            device_num = 1
            data_file = one_data_info["data_path"]
            data_name = one_data_info["data_name"]
            # output_file is the combination of slice num and dataset name and model name
            # we assume one instance process one model and one dataset
            with open(data_file, "r") as f:
                data_lines = f.readlines()

            data_lines_length = len(data_lines)
            
            num_process_models = one_model_info["num_gpu"]
            # Calculate the number of tasks to assign to each process model
            num_partitions = data_lines_length // (partition_size * num_process_models)
            if data_lines_length % (partition_size * num_process_models) > 0:
                num_partitions += 1

            for idx_partition in range(num_partitions):
                start_idx = idx_partition * partition_size * num_process_models
                end_idx = min(
                    start_idx + partition_size * num_process_models, data_lines_length
                )

                if idx_partition == num_partitions - 1:
                    end_idx += 1

                inner_data_lines_length = end_idx - start_idx
                task_count_per_model_per_instance = (
                    inner_data_lines_length // num_process_models
                )
                remainder_tasks_inner = inner_data_lines_length % num_process_models

                task_ranges_inner = []

                start_idx_inner = start_idx
                for i_num_process in range(num_process_models):
                    end_idx_inner = start_idx_inner + task_count_per_model_per_instance
                    if i_num_process < remainder_tasks_inner:
                        end_idx_inner += 1  # Distribute the remainder tasks equally among the first 'remainder_tasks' process models
                    task_ranges_inner.append((start_idx_inner, end_idx_inner))
                    start_idx_inner = end_idx_inner

                start_index_list = (
                    "("
                    + " ".join([str(start_i) for start_i, end_i in task_ranges_inner])
                    + ")"
                )
                end_index_list = (
                    "("
                    + " ".join([str(end_i) for start_i, end_i in task_ranges_inner])
                    + ")"
                )

                gpu_id_assignment = []

                if num_process_models == 8:
                    gpu_id_assignment = [str(i) for i in range(8)]
                elif num_process_models == 4:
                    for i in range(0, 8, 2):
                        gpu_id_assignment.append(f"{i},{i+1}")
                else:
                    # Handle other cases as needed
                    pass
                print(gpu_id_assignment)
                row_gpu_ids = "("
                for one_id_tuple in gpu_id_assignment:
                    row_gpu_ids += '"' + one_id_tuple + '"' + " "
                row_gpu_ids = row_gpu_ids.strip()
                row_gpu_ids += ")"
                log_dir_model_data = os.path.join(log_file, model_name, data_name)
                os.makedirs(log_dir_model_data, exist_ok=True)

                output_dir_model_data = os.path.join(output_dir, model_name, data_name)
                os.makedirs(output_dir_model_data, exist_ok=True)

                script_folder = os.path.join(where_to_save_shell, model_name)
                os.makedirs(script_folder, exist_ok=True)
                script_file_name = os.path.join(
                    script_folder, "{}_part{}.sh".format(data_name, idx_partition)
                )
                
                script_file_names.append(script_file_name)  # Save the script file name

                with open(script_file_name, "w") as script_file:
                    script_file.write("#!/bin/bash\n\n")
                    script_file.write(
                        f"pip install transformers_stream_generator -i https://pypi.tuna.tsinghua.edu.cn/simple\n"
                    )
                    script_file.write(
                        f"pip install -U transformers -i https://pypi.tuna.tsinghua.edu.cn/simple\n"
                    )

                    script_file.write(f'output_dir="{output_dir_model_data}"\n')
                    script_file.write(f"gpu_ids={row_gpu_ids}\n")
                    script_file.write(f"model_path={model_path}\n")
                    script_file.write(f"model_name={model_name}\n")
                    script_file.write(f"data_name={data_name}\n")
                    script_file.write(f"data_file={data_file}\n")
                    script_file.write(f"device_num={device_num}\n")
                    script_file.write(f"log_dir_model_data={log_dir_model_data}\n")
                    script_file.write(f"start_index_list={start_index_list}\n")
                    script_file.write(f"end_index_list={end_index_list}\n")

                    script_file.write(f'mkdir -p "$output_dir"\n')
                    script_file.write(
                        f"pids=() # Array to store the PIDs of the background processes\n\n"
                    )

                    script_file.write("for ((i=0; i<${#gpu_ids[@]}; i++)); do\n")
                    script_file.write("    gpu_id=${gpu_ids[$i]}\n")
                    script_file.write("    start_index=${start_index_list[$i]}\n")
                    script_file.write("    end_index=${end_index_list[$i]}\n")

                    script_file.write(
                        f'    echo "开始评估模型: $model_name, GPU ID: $gpu_id, Start index: $start_index, End index: $end_index"\n'
                    )

                    script_file.write(
                        f'    output_file="$output_dir/output_${{model_name}}_${{data_name}}_${{start_index}}-${{end_index}}_gpu-${{gpu_id}}.json"\n'
                    )

                    script_file.write(
                        f'    log_file="$log_dir_model_data/${{model_name}}_${{data_name}}_${{start_index}}-${{end_index}}_gpu-${{gpu_id}}.log"\n\n'
                    )

                    script_file.write(
                        '    command="CUDA_VISIBLE_DEVICES=$gpu_id nohup python -u inference_matheval.py'
                    )  # 改inference的python文件
                    if "3shot" in data_name:
                        script_file.write(f" --few_shot")
                    script_file.write(
                        f" --model_name $model_name --model_path $model_path --data_file $data_file"
                    )
                    script_file.write(
                        f' --output_file $output_file --start_index $start_index --end_index $end_index --gpu_id $gpu_id --device_num $device_num --max_new_tokens 512 > $log_file 2>&1 &"\n\n'
                    )

                    script_file.write('    eval "$command"\n')
                    script_file.write(
                        "    pids+=($!) # Store the PID of the last command run in the background\n\n"
                    )
                    script_file.write("done\n")
                    script_file.write('for pid in "${pids[@]}"; do\n')
                    script_file.write('    wait "$pid"\n')
                    script_file.write("done\n\n")

                    script_file.write(
                        "# Kill all processes related to evaluating the model no api\n"
                    )
                    script_file.write("pkill -f inference_matheval.py\n\n")  # 改kill的名字

                    script_file.write(
                        'echo "All evaluations completed and related processes killed."\n'
                    )

                # Make the script executable
                os.chmod(script_file_name, 0o755)
    
    return script_file_names  # Return the list of script file names