# MathEval

MathEval is a benchmark dedicated to a comprehensive evaluation of the mathematical capabilities of large models. It encompasses over 20 evaluation datasets across various mathematical domains, with over 30,000 math problems. The goal is to thoroughly evaluate the performance of large models in tackling problems spanning a wide range of difficulty levels and diverse mathematical subfields (i.e. arithmetic, elementary mathematics, middle and high school competition topics, advanced mathematical, etc.). It serves as the trustworthy reference for cross-model comparisons of mathematical abilities among large models at the current stage and provides guidance on how to further enhance the mathematical capabilities of these models in the future.


# Evaluation Pipeline
**The whole pipeline is time consuming, we recommend to submit your evaluation request through our [website](https://matheval.ai/)**

## Step0: Download Dataset
Download the evaluation datasets from [Google Drive](https://drive.google.com/file/d/1_fhRIXwpHv0lGtcVShKQOKiy4ITp-ttB/view?usp=sharing)

## Step1: Dataset preprocessing
We unify the input format of different mathematical dataset as following:

For zero shot settings:
```json
{"conversations": [{"from":"human", "value":"xxx"}]}
```

For few shot settings:
```json
{"conversations": [{"from":"human", "value":"example0-question"}, {"from":"gpt", "value":"example0-answer"}, {},{}, ...]}
```

Running Script:

```shell
python preprocess_dataset/build_dataset.py --input_dir ./datasets --output_dir ./output
```
where ./datasets is the path where you save your downloaded datasets

## Step2: How to construct the input prompt and the inference script 
We format the input prompt for different base model and their chat edition based on the [fastchat](https://github.com/lm-sys/FastChat/tree/main/fastchat).

- If you have elaborate templates for your models, implement your **templates** inside the conversation.py file, and change the settings in generate_shell_config.py file with your own **template_name**; 
- You can also use default template_name for your model. Please also change the **data_dir** variable in generate_shell_config.py file. The generate_shell_config.py contains all the preprocessing config for each dataset and each base model.
- We assume you have at least 2 80G-GPUs in your device to run over 70B size models. You need to ensure that the model can be successfully loaded with the given num_gpus. Please run the following script to generate the inference scripts, you need to specify the path to the model's output directory, the path to the log directory, and the path to save the running shell scripts:


## Step3: How to Run your model with the processed dataset

### Basic Run Command
First, use the following command to run your model:
```
python run.py --configs ./generate_shell_config.py
```

### Using VLLM for Inference Acceleration
We now support using VLLM for inference acceleration. To enable VLLM acceleration, use the following command:
```
python run.py --configs ./generate_shell_config.py --accelerator vllm
```

### Installing VLLM
To use VLLM for inference acceleration, you need to install it first. Please refer to the [VLLM GitHub repository](https://github.com/vllm-project/vllm) for installation instructions.

### Parameter Explanation
- --configs ./generate_shell_config.py: Specifies the path to the configuration file.
- --accelerator vllm: (Optional) Enables VLLM for inference acceleration.

## Step 4: Compare-Answer
If you have access to GPT4, please jump to Step 5.

If you do not have access to GPT4, we have released a compare-answer model in HuggingFace, please refer to:
[DeepSeek-Math-Compare-Answer](https://huggingface.co/Tianqiao/DeepSeek-7B-Math-Compare-Answer)

Now we implement compare compare_with_local_model.py
###Example Command
```
python compare_with_local_model_hg.py --model_path /path/to/model --input_dir /path/to/input --output_dir /path/to/output --device_num 4
```

###Command Line Arguments
- --model_path: Path to the pre-trained model directory.
- --input_dir: Directory containing the input JSON files.
- --output_dir: Directory to save the output JSON files.
- --device_num: Number of GPUs to use for parallel processing.

## Step5: Answer-compare with GPT4

### How to extract the answer
We have provide our prompt in ./prompts/extraction_prompts folder.

We recommend to realize the send_chat_request function in run_gpt4_extraction.py by yourself, since we do not know the detailed GPT4 invoking function of yours.

### How to verify the answer generated from GPT4 and the answer from the golden dataset

We have provide our prompt in ./prompts/verification_prompts folder.

We recommend to realize the send_chat_request function in run_gpt4_extraction.py by yourself, since we do not know the detailed GPT4 invoking function of yours.


# Discussion
Why we want to use GPT4 for answer extraction and answer verification.

## precision comparison
The precision comparison of using GPT4 and REGEX from OpenCompass for answer extractor
![Figure 1](/figures/extraction_comparison.png "GPT4 and REGEX from OpenCompass for answer extractor")

## Corner cases for extraction
![Figure 2](/figures/corner_case_extraction_0.png "extraction comparison 0")

![Figure 3](/figures/corner_case_extraction_1.png "extraction comparison 1")

![Figure 4](/figures/corner_case_extraction_2.png "extraction comparison 2")

![Figure 5](/figures/corner_case_extraction_3.png "extraction comparison 3")

## The precision comparison of using GPT4 and REGEX from OpenCompass for answer verification
![Figure 6](/figures/verification_comparison.png "verification comparison")

![Figure 7](/figures/corner_case_verification_0.png "verification comparison 0")

![Figure 8](/figures/corner_case_verification_1.png "verification comparison 1")

![Figure 9](/figures/corner_case_verification_2.png "verification comparison 2")

![Figure 10](/figures/corner_case_verification_3.png "verification comparison 3")

# Our total cost
![Figure 11](/figures/cost.png "cost")

# Dataset
| Dataset_name | Dataset_size |
| ------ | ---- |
| agieval-aqua-rat | 254 | 
| agieval-gaokao-mathcloze | 118| 
| agieval-gaokao-mathqa | 351| 
| agieval-math | 1000| 
| agieval-ppl-aqua-rat | 254| 
| agieval-ppl-gaokao-mathcloze | 118| 
|agieval-ppl-gaokao-mathqa | 351| 
|agieval-ppl-math | 1000| 
|agieval-ppl-sat-math | 220| 
|agieval-sat-math | 220| 
|ape210k | 5000| 
|ape210k-few3 | 5000| 
|asdiv-a | 122 | 
|asdiv-a-few | 122 | 
|bbh-fewmultistep_arithmetic_two | 250 | 
|bbh-multistep_arithmetic_two | 250 | 
|ceval-few-test-advanced_mathematics | 173 | 
|ceval-few-test-discrete_mathematics | 153 | 
|ceval-few-test-high_school_mathematics | 166 | 
|ceval-few-test-middle_school_mathematics | 177 | 
|ceval-few-test-probability_and_statistics | 166 | 
|ceval-ppl-test-advanced_mathematics | 173 | 
|ceval-ppl-test-discrete_mathematics | 153 | 
|ceval-ppl-test-high_school_mathematics | 166 | 
|ceval-ppl-test-middle_school_mathematics | 177 | 
|ceval-ppl-test-probability_and_statistics | 166 | 
|ceval-test-advanced_mathematics | 173 | 
|ceval-test-discrete_mathematics | 153 | 
|ceval-test-high_school_mathematics | 166 | 
|ceval-test-middle_school_mathematics | 177 | 
|ceval-test-probability_and_statistics | 166 | 
|cmmlu-college_mathematics | 105 | 
|cmmlu-elementary_mathematics | 230 | 
|cmmlu-few-college_mathematics | 105 | 
|cmmlu-few-elementary_mathematics | 230 | 
|cmmlu-few-high_school_mathematics | 164 | 
|cmmlu-high_school_mathematics | 164 | 
|dolphin1878 | 187 | 
|dolphin1878-few | 187 | 
|draw | 200 | 
|draw-few | 200 | 
|GaokaoBench_2010-2022_Math_I_MCQs | 214 | 
|GaokaoBench_2010-2022_Math_II_MCQs | 218 | 
|GaokaoBench_few2010-2022_Math_I_MCQs | 214 | 
|GaokaoBench_few2010-2022_Math_II_MCQs | 218 | 
|gsm8k | 1319 | 
|gsm8k-few | 1319 | 
|hmwp | 550 | 
|hmwp-few | 550 | 
|lukaemon_mmlu_abstract_algebra | 100 | 
|lukaemon_mmlu_college_mathematics | 100 | 
|lukaemon_mmlu_elementary_mathematics | 378 | 
|lukaemon_mmlu_fewabstract_algebra | 100 | 
|lukaemon_mmlu_fewcollege_mathematics | 100 | 
|lukaemon_mmlu_fewelementary_mathematics | 378 | 
|lukaemon_mmlu_fewhigh_school_mathematics | 270 | 
|lukaemon_mmlu_high_school_mathematics | 270 | 
|math | 5000 | 
|math23k | 2317 | 
|math23k-few5 | 2317 | 
|math401 | 401 | 
|math401-few | 401 | 
|math-few | 5000 | 
|MathQA | 2985 | 
|MathQA-few | 2985 | 
|MathQA-ppl | 2985 | 
|mawps | 238 | 
|mawps-few | 238 | 
|mmlu_ppl_fewabstract_algebra | 100 | 
|mmlu_ppl_fewcollege_mathematics | 100 | 
|mmlu_ppl_fewelementary_mathematics | 378 | 
|mmlu_ppl_fewhigh_school_mathematics | 270 | 
|mmlu_pplabstract_algebra | 100 | 
|mmlu_pplcollege_mathematics | 100 | 
|mmlu_pplelementary_mathematics | 378 | 
|mmlu_pplhigh_school_mathematics | 270 | 
|scq_ch | 2000 | 
|scq_ch_few | 2000 | 
|scq_en | 2000 | 
|scq_en_few | 2000 | 
|svamp | 1000 | 
|svamp-few | 1000|


