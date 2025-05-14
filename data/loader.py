from datasets import load_dataset, concatenate_datasets, Dataset
from data.constants import BIGBENCH_SUBTASKS
from utils.answer_extraction import extract_gsm8k, extract_math
from utils.seeds import initialize_seeds
import string
import numpy as np
import re

def load_data(dataset, optim=False):
    initialize_seeds()
    if dataset == "mmlu_pro":
        split = "test" if not optim else "validation"
        return load_dataset("TIGER-Lab/MMLU-Pro")[split]
    elif dataset == "truthfulqa":
        if optim: pass
        return load_dataset("truthfulqa/truthful_qa", "multiple_choice")["validation"]
    elif dataset == "bigbench":
        bigbench = []
        for task in BIGBENCH_SUBTASKS:
            split = "validation" if not optim else "train"
            n_samples = 1000 if not optim else 100
            dataset = load_dataset("tasksource/bigbench",task, trust_remote_code=True)
            dataset[split] = dataset[split].add_column("category", dataset[split].num_rows*[task])
            bigbench.append(dataset[split].select(range(min(dataset[split].num_rows, n_samples))))
        return concatenate_datasets(bigbench)
    elif dataset == "gsm8k":
        split = "test" if not optim else "train"
        dataset = load_dataset("openai/gsm8k", "main")[split]
        if optim: 
            dataset = dataset.select(range(min(dataset.num_rows, 100)))
        return dataset
    elif dataset == "math": 
        if not optim:
            return load_dataset("lighteval/MATH", "all")["test"]
        else:
            dataset = load_dataset("DigitalLearningGmbH/MATH-lighteval", "default")["test"]
            df = dataset.to_pandas()
            df = df.groupby("type").sample(50)
            return Dataset.from_pandas(df, preserve_index=False)
    else:
        raise NotImplementedError
    
def strip_empty_lines(text):
    return re.sub(r'\n+', '\n', text)
    
def format_mmlu_pro(examples, expertise):
    alpha = string.ascii_uppercase
    inputs = []
    for instruction, options in zip(examples["question"], examples["options"]):
        choice_string = "\n".join([f"({letter}) {choice}" for letter, choice in zip(alpha, options)])
        instruction = f"{instruction}\n{choice_string}"
        if expertise: instruction = strip_empty_lines(instruction)
        input = [
                {
                    "role": "user",
                    "content": instruction,
                },
                ]
        inputs.append(input)
    return {"prompt": inputs}

def format_bigbench(examples, expertise):
    alpha = string.ascii_uppercase
    inputs = []
    labels = []
    np.random.seed(42)
    for instruction, choices, scores in zip(examples["inputs"], examples["multiple_choice_targets"], examples['multiple_choice_scores']):
        choices = np.array(choices)
        if type(scores) == list: scores = np.array(scores)
        if len(choices) > 26:
            ii = np.where(scores == 0)[0]
            remove = np.random.choice(ii, size=len(choices) - 26, replace=False)
            choices = np.delete(choices, remove)
            scores = np.delete(scores, remove)
        indices = np.arange(len(choices))
        np.random.shuffle(indices)
        choices = choices[indices]
        scores = scores[indices]
        labels.append(alpha[list(scores).index(1)])
        choice_string = "\n".join([f"({letter}) {choice}" for letter, choice in zip(alpha, choices)])
        instruction = instruction.split("\n  choice: ")[0]
        instruction = f"{instruction}\n{choice_string}"
        if expertise: instruction = strip_empty_lines(instruction)
        input = [
            {
                "role": "user",
                "content": instruction,
            },
            ]
        inputs.append(input)
    return {"prompt": inputs, "label": labels}
    
def format_truthfulqa(examples, expertise):
    alpha = string.ascii_uppercase
    inputs = []
    labels = []
    np.random.seed(42)
    for instruction, targets in zip(examples["question"], examples["mc1_targets"]):
        choices = targets["choices"]
        choices = np.array(choices)
        scores = targets["labels"]
        scores = np.array(scores)
        indices = np.arange(len(choices))
        np.random.shuffle(indices)
        choices = choices[indices]
        scores = scores[indices]
        labels.append(alpha[list(scores).index(1)])
        choice_string = "\n".join([f"({letter}) {choice}" for letter, choice in zip(alpha, choices)])
        instruction = f"{instruction}\n{choice_string}"
        if expertise: instruction = strip_empty_lines(instruction)
        input = [
                {
                    "role": "user",
                    "content": instruction,
                },
                ]
        inputs.append(input)
    return {"prompt": inputs, "label": labels}

def format_gsm8k(examples, expertise):
    inputs = []
    labels = []
    for instruction, answer in zip(examples["question"], examples["answer"]):
        instruction = f"Read the question below and provide the final numerical answer in a separate line at the end.\n\n{instruction}" if not expertise else instruction
        if expertise: instruction = strip_empty_lines(instruction)
        input = [
                {
                    "role": "user",
                    "content": instruction,
                },
                ]
        inputs.append(input)
        labels.append(extract_gsm8k(answer))
    return {"prompt": inputs, "label": labels}

def format_math(examples, expertise):
    inputs = []
    labels = []
    for instruction, answer in zip(examples["problem"], examples["solution"]):
        instruction = f"Read the question below and provide the final answer in a LaTeX boxed environment.\n\n{instruction}" if not expertise else instruction
        if expertise: instruction = strip_empty_lines(instruction)
        input = [
                {
                    "role": "user",
                    "content": instruction,
                },
                ]
        inputs.append(input)
        labels.append(extract_math(answer))
    return {"prompt": inputs, "label": labels}
    
def format_prompt(dataset):
    if dataset == "mmlu_pro": return format_mmlu_pro
    elif dataset == "truthfulqa": return format_truthfulqa
    elif dataset == "bigbench": return format_bigbench
    elif dataset == "gsm8k": return format_gsm8k
    elif dataset == "math": return format_math
    else: raise NotImplementedError
    
def process_dataset(dataset, name, expertise=False):
    dataset = dataset.map(lambda x: format_prompt(name)(x, expertise), batched=True)
    return dataset