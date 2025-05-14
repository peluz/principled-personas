import argparse
from data.constants import DATASETS
from pathlib import Path
from utils.seeds import initialize_seeds
from tqdm.auto import tqdm
from vllm.sampling_params import SamplingParams
from data.personas import *
from data.persona_templates import persona_to_desc, mitigation_template
from data.expertise_prompts import expertise_prompts
from data.loader import load_data, process_dataset
import json
import os
import pandas as pd
import random


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(llm, model_url, expertise, from_expertise, mitigate, dataset, personas_path, temperature, top_p):
    initialize_seeds()
    sampling_params = SamplingParams(max_tokens=4096, temperature=temperature)
    if "gemma" in model_url:
        has_system = False
    else:
        has_system = True
    if personas_path is None:
        personas = EDUCATION_PERSONAS + NO_PERSONA + NAMES + COLOR_PERSONAS
    else:
        personas = json.load(open(personas_path, "r"))["personas"]
    dataset_list = DATASETS if "all" in dataset else dataset
    datasets = {}
    path = f"./results/{mitigate}" if mitigate is not None else "./results"
    result_folder = Path(path)
    for dataset in dataset_list:
            model_name = model_url.split("/")[-1]
            print(f"Loading dataset: {dataset}")
            data = load_data(dataset)
            datasets[dataset] = process_dataset(data, dataset, expertise)
    
    for dataset_name in datasets.keys():
        if expertise:
            result_path = result_folder/"expertise"/f"{dataset_name}.csv"
            if result_path.exists(): continue
            initialize_seeds()
            print(f"Generating expertises for dataset {dataset_name}.\n\n")
            prompts = datasets[dataset_name]["prompt"]
            task =  expertise_prompts[dataset_name]
            generating_prompts = [[{ "role": "system", "content": task}] + p if has_system else [{"role": "user", "content": f"{task}\n\n{p[-1]['content']}\nAnswer:"}] for p in prompts]
            print(f"Example prompt: {random.choice(generating_prompts)[-1]['content']}")
            llm.chat(generating_prompts[0], sampling_params=sampling_params)
            outputs = llm.chat(generating_prompts, sampling_params=sampling_params)
            responses = []
            for output in outputs:
                generated_text = output.outputs[0].text
                responses.append(generated_text)
            result_path.parent.mkdir(exist_ok=True, parents=True)
            pd.DataFrame({model_name: responses}).to_csv(result_path, index=False)
        elif from_expertise:
            expertise_df = pd.read_csv(f"./data/experts/{dataset_name}.csv")
            for level in tqdm(["level1", "level2", "level3"]):
                result_path = result_folder/model_name/dataset_name/f"{level}.csv"
                if result_path.exists(): continue
                initialize_seeds()
                print(f"Generating responses using model {model_url} for dataset {dataset_name} and {level} experts.\n\n")
                prompts = datasets[dataset_name]["prompt"]
                personas = expertise_df[level].tolist()
                if mitigate: personas = [mitigation_template[mitigate].format(persona_desc=x) for x in personas]
                if mitigate not in ["refine", "refine_basic"]:
                    generating_prompts = [[{ "role": "system", "content": f"{persona}."}] + p if has_system else [{"role": "user", "content": f"{persona}.\n\n{p[-1]['content']}"}] for p, persona in zip(prompts, personas)]
                    print(f"Example prompt: {random.choice(generating_prompts)}")
                else:
                    original_responses = pd.read_csv(f"./results/{model_name}/{dataset_name}/empty.csv").iloc[:,-1].tolist()
                    generating_prompts = [[{"role": "user", "content": f"{p[-1]['content']}"},
                                           {"role": "assistant", "content": r},
                                           {"role": "user", "content": persona}] for p, persona, r in zip(prompts, personas, original_responses)]
                    print(f"Example prompt: {random.choice(generating_prompts)}")
                llm.chat(generating_prompts[0], sampling_params=sampling_params)
                outputs = llm.chat(generating_prompts, sampling_params=sampling_params)
                responses = []
                for output in outputs:
                    generated_text = output.outputs[0].text
                    responses.append(generated_text)
                result_path.parent.mkdir(exist_ok=True, parents=True)
                pd.DataFrame({model_name: responses}).to_csv(result_path, index=False)
        else:
            for persona in tqdm(personas + EXPERTS[dataset_name]):
                persona_name = persona.replace(' ', '_') if persona != "" else "empty"
                if persona_name == "empty" and mitigate: continue
                result_path = result_folder/model_name/dataset_name/f"{persona_name}.csv"
                if result_path.exists(): continue
                initialize_seeds()
                print(f"Generating responses using model {model_url} for dataset {dataset_name} and persona {persona}\n\n")
                prompts = datasets[dataset_name]["prompt"]
                description =  persona_to_desc(persona)
                if description is not None:
                    if mitigate: description = mitigation_template[mitigate].format(persona_desc=description)
                    if mitigate not in ["refine", "refine_basic"]:
                        generating_prompts = [[{ "role": "system", "content": f"{description}."}] + p if has_system else [{"role": "user", "content": f"{description}.\n\n{p[-1]['content']}"}] for p in prompts]
                    else:
                        original_responses = pd.read_csv(f"./results/{model_name}/{dataset_name}/empty.csv").iloc[:,-1].tolist()
                        generating_prompts = [[{"role": "user", "content": f"{p[-1]['content']}"},
                                            {"role": "assistant", "content": r},
                                            {"role": "user", "content": description}] for p, r in zip(prompts, original_responses)]
                else: generating_prompts = prompts
                print(f"Example prompt: {random.choice(generating_prompts)}")
                llm.chat(generating_prompts[0], sampling_params=sampling_params)
                outputs = llm.chat(generating_prompts, sampling_params=sampling_params)
                responses = []
                for output in outputs:
                    generated_text = output.outputs[0].text
                    responses.append(generated_text)
                result_path.parent.mkdir(exist_ok=True, parents=True)
                pd.DataFrame({model_name: responses}).to_csv(result_path, index=False)
if __name__ == '__main__':
    import vllm
    parser = argparse.ArgumentParser(
        description='Get the predictions for a given model.')
    parser.add_argument('model_url', help='The model to be prompted.', type=str)
    parser.add_argument('--expertise', help="Generate expertises rather than responses",action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--from_expertise', help="Generate responses from precomputed expertises",action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--mitigate', help="Use a mitigation strategy",type=str, choices=list(mitigation_template.keys()), default=None)
    parser.add_argument('--dataset', help='The datasets to be processed.', nargs="+", type=str, choices=DATASETS + ["all"], default=["all"])
    parser.add_argument('--personas_path', help='A path to file with the personas to be induced.', type=str, default=None)
    parser.add_argument('--gpus', help='Number of gpus', type=int, default=1)
    parser.add_argument('--temperature', help='Temperature for probabiliy scaling.', type=float, default=0.0)
    parser.add_argument('--top_p', help='Top-p proability of tokens for nucleus sampling', type=float, default=1.0)
    parser.add_argument('--dtype', help='dtype to load the model', type=str, default="auto")
    args = parser.parse_args()
    llm = vllm.LLM(model=args.model_url, enable_prefix_caching=True, dtype=args.dtype, tensor_parallel_size=args.gpus,
            #   download_dir=os.environ["HF_MODELS"],
              gpu_memory_utilization=.95)
    main(llm, args.model_url, args.expertise, args.from_expertise, args.mitigate, args.dataset, args.personas_path, args.temperature, args.top_p)