# MathEval

MathEval is a benchmark dedicated to a comprehensive evaluation of the mathematical capabilities of large models. It encompasses over 20 evaluation datasets across various mathematical domains, with over 30,000 math problems. The goal is to thoroughly evaluate the performance of large models in tackling problems spanning a wide range of difficulty levels and diverse mathematical subfields (i.e. arithmetic, elementary mathematics, middle and high school competition topics, advanced mathematical, etc.). It serves as the trustworthy reference for cross-model comparisons of mathematical abilities among large models at the current stage and provides guidance on how to further enhance the mathematical capabilities of these models in the future.


# Evaluation Pipeline
## Download Dataset
Download the evaluation datasets from [Google Drive](https://drive.google.com/file/d/1_fhRIXwpHv0lGtcVShKQOKiy4ITp-ttB/view?usp=sharing)

## Dataset preprocessing
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
python preprocess_dataset/build_dataset.py --input_dir ./datasets --output_dir ./output


## How to construct the input prompt
We format the input prompt for different base model and their chat edition based on the [fastchat](https://github.com/lm-sys/FastChat/tree/main/fastchat).

If you have elaborate templates for your models, implement your templates inside the conversation.py file, and change the settings in generate_shell_config.py file with your own template_name; You can also use default template_name for your model. Please also change the data_dir variable in generate_shell_config.py file.

## How to Run your model with the processed dataset





# Dataset
| 数据集名字（包含子数据名字） | 数据集大小 |
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



# 大模型

# 评测框架

