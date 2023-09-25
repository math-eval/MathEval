### Arith_std_30K
   
    dict(  
    role="HUMAN",  
    prompt="Calculate the following math word problem: {expression}\n Answer:"  
    )


### TAL-SCQ5K-EN

    dict(  
    role="HUMAN",  
    prompt="Solve the following math word problem and choose a final choice among the provided multiple choices A,B,C,D,E : {problem}\n, Answer:"  
    )


### TAL-SCQ5K-CN

    dict(  
    role="HUMAN",  
    prompt="计算以下数学单选题并给出最终选项 A,B,C,D,E 之一: {problem}\n, 答案:"  
    )


### MAWPS

    dict(  
    role="HUMAN",  
    prompt="Calculate the following math word problem: {original_text}\n Answer:"  
    )


### GAOKAO(Math)

    {  
    "role": "HUMAN",  
    "prompt": "请你做一道数学选择题\n请你一步一步思考并将思考过程写在【解析】和<eoe>之间。你将从A，B，C，D中选出正确的答案，并写在【答案】和<eoa>之间。\n例如：【答案】: A <eoa>\n完整的题目回答的格式如下：\n【解析】 ... <eoe>\n【答案】 ... <eoa>\n请你严格按照上述格式作答。\n题目如下：' + '{question}'  
    }

    
###  ASDiv-A

    dict(  
    role="HUMAN",  
    prompt=  
    "Calculate the following math word problem: {problem}\n Answer:"  
    )

### CMMLU(Math)

    dict(  
    role="HUMAN",  
    prompt=f"以下是关于{_ch_name}的单项选择题，请直接给出正确答案的选项。\n题目：{{question}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}"  
    ),  
    dict(role="BOT", prompt='答案是: {answer}')    


###  MATH

    dict(role="HUMAN", prompt="Problem:\n{problem}\nSolution:\n")

    
### GSM8k
    
    dict(role='HUMAN', prompt="Calculate the following math word problem: {question}\nPlease give the answer directly\nAnswer:")

### MMLU(Math)
    dict(  
    role="HUMAN",  
    prompt=f"There is a single choice question:\nQuestion: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nPlease give your answer directly from the four options A, B, C, D: "  
    )
    
### SVAMP
    dict(role='HUMAN', prompt="Calculate the following math word problem: {problem}\nPlease give the answer directly\nAnswer:")

### math401-llm
    dict(role='HUMAN', prompt="Calculate the following math problem: {problem}\nPlease give the answer directly\nAnswer:")
    
### DRAW1K
    dict(role='HUMAN', prompt="Calculate the following math problem: {problem}\nPlease give the answer directly\nAnswer:")
    
### HMWP
    dict(role='HUMAN', prompt="回答下面的数学题:\n{problem}\n请直接给出最终答案:\n")
    
### Dolphin1878
    dict(role='HUMAN', prompt="Calculate the following math problem: {problem}\nPlease give the answer directly\nAnswer:")
    
### C-eval(Math)
    dict(  
    role="HUMAN",  
    prompt=f"以下是中国关于{_ch_name}考试的单项选择题，请选出其中的正确答案。\n{{question}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\n直接给出答案，答案是: "  
    )

### MathQA
    dict(  
    role='HUMAN',  
    prompt='Here is a math question, you need to choose the correct answer from the options based on the content of the question. Problem:{Problem}\n,options:{options}\n,The choose the correct option is:'  
    )
    
### Math23K
    dict(role="HUMAN",prompt="下面是一个数学题，根据问题回答，直接给出答案即可。问题:{original_text}\n直接给出答案:")
    
### Ape210K
    dict(  
    role='HUMAN',  
    prompt='下面是一个数学题，根据问题回答，直接给出答案即可。问题:{original_text}\n直接给出答案:'  
    )
    
### Big-Bench(Math)
    dict(  
    role="HUMAN",  
    prompt=f"Here is a math question, you need answer the question. Question:{{input}}\n,The correct answer is ",  
    )
    
### Big-Bench-Hard(Math)
    dict(  
    role="HUMAN",  
    prompt=f"Follow the given examples and answer the question.\nQ: {{input}}\nA: Let's think step by step."  
    )
    
### AGIEval
    dict(  
    role='HUMAN', prompt=f'{{question}}\n{{options}}\n答案是： ')  
    )

