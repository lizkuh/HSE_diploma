# exp_name = "t2c_concode_220428_v33"
import fire

import sys 
sys.path.append("/root/HSE_diploma/")

import pickle
from datetime import datetime


import numpy as np
import json
import pandas as pd
import os

from transformers import LlamaForCausalLM as LLaMAForCausalLM
from transformers import LlamaTokenizer as LLaMATokenizer

from peft import PeftModel
import torch
from prompter import generate_test_prompt, get_response
from transformers import GenerationConfig
import math
import tqdm
import re


def preprocess(s):
    #ToDo rewrite it using Promt Template
    s = get_response(s)
    if "</s>" in s:
        s = s.split('</s>')[0]
    if "### Input" in s:
        s = s.split("### Input")[0]
    try:
        assert "<unk>" not in s
        assert "\n" not in s
        assert "  " not in s
    except Exception as e:
        print("Something wrong with")
        print(s)
        raise e
    return s


# "/root/temperary_results/predictionst2c_concode_220428_v33_2023-05-26-16:46:49.txt"
def run_evaluation_of_metrics(fn):
    #fn_refference = "/root/data/reference_corpus.txt"
    #fn_answers = "/root/data/answers.json"
    
    fn_refference = "/root/data/final_dev.txt"
    fn_answers = "/root/data/final_answers_jsonl.json"
    open("/root/HSE_diploma/load_all_experiments/res.txt", "w+").write('')
    print(f"/root/HSE_diploma/evaluateFromCodeXGlue/;python3 calc_code_bleu.py --refs {fn_refference} --hyp $fn --lang java --params 0.25,0.25,0.25,0.25")
    os.system("cd /root/HSE_diploma/evaluateFromCodeXGlue/;python3 calc_code_bleu.py --refs {fn_refference} --hyp $fn --lang java --params 0.25,0.25,0.25,0.25 > /root/HSE_diploma/load_all_experiments/res.txt")
    os.system("cd /root/HSE_diploma/evaluateFromCodeXGlue/;python3 calculate_bleu/evaluator.py -a={fn_answers} -p=$fn  >> /root/HSE_diploma/load_all_experiments/res.txt")


def parse_metrics():
    s = open('res.txt', "r").read()

    BLEU = re.findall("BLEU: \d+\.\d+", s)
    EM = re.findall("EM: \d+\.\d+", s)
    CodeBLEU = re.findall("CodeBLEU score:  \d+\.\d+", s)

    assert len(BLEU) == len(EM) == len(CodeBLEU) == 1
    BLEU = float(BLEU[0].split(' ')[-1])
    EM = float(EM[0].split(' ')[-1])
    CodeBLEU = float(CodeBLEU[0].split(' ')[-1])

    ngram_match = float(re.findall("ngram match: \d+\.\d+", s)[0].split(' ')[-1]) 
    weighted_ngram_match = float(re.findall("weighted ngram match: \d+\.\d+", s)[0].split(' ')[-1]) 
    syntax_match = float(re.findall("syntax_match: \d+\.\d+", s)[0].split(' ')[-1]) 
    dataflow_match = float(re.findall("dataflow_match: \d+\.\d+", s)[0].split(' ')[-1]) 

    precisions = re.findall("precisions:\s+.+\n", s)[0].split(':  ')[-1][:-1]
    bp = float(re.findall("bp:  \d+\.\d+", s)[0].split(' ')[-1])
    ratio = float(re.findall("ratio:  \d+\.\d+", s)[0].split(' ')[-1])
    translation_length = int(re.findall("translation_length:.*\n", s)[0].split(' ')[-1][:-1])
    reference_length = int(re.findall("reference_length:.*\n", s)[0].split(' ')[-1][:-1])

    # weighted ngram match: 0.2792407600329364, 
    # syntax_match: 0.3445378151260504, 
    # dataflow_match: 0.2908777969018933
    metrics = {"BLEU": BLEU,
               "EM": EM,
               "CodeBLEU": CodeBLEU,
               "ngram_match": ngram_match,
               "weighted_ngram_match": weighted_ngram_match,
               "syntax_match": syntax_match,
               "dataflow_match": dataflow_match,
               "precisions": precisions,
               "bp": bp,
               "ratio": ratio,
               "translation_length": translation_length,
               "reference_length": reference_length,
               "raw": s
              }
    return metrics


def evaluate(exp_name: str):
    """
         Load model from name node and evaluate metrics on full dataset
    """
    EXPERIMENT_PATH = "/root/experiments/"
    ARTIFACTS_PATH = "/root/temperary_results/"

    df_experiments = pd.read_csv("all_experiments.csv").set_index("experiment_name_short")
    df_experiments.loc[["9", "10", "11", "12", "13"], "default_model"] = ["decapoda-research/llama-7b-hf"]*5

    line = df_experiments[df_experiments['exp_name'] == exp_name]
    assert(len(line)==1)
    line = line.iloc[0]
    line = line.to_dict()

    experiment_name = line["experiment_name"]
    if not isinstance(experiment_name, str):
        experiment_name = exp_name
    
    print("experiment_name", experiment_name)
    # Loading files to remote host if they are not exist
    if not os.path.exists(os.path.join(EXPERIMENT_PATH, experiment_name)):
        print("loading files")
        #!mkdir {os.path.join(EXPERIMENT_PATH, experiment_name)}
        os.system(f"mkdir {os.path.join(EXPERIMENT_PATH, experiment_name)}")
        for filename in ['adapter_model.bin', 'experiment_config.json', 'adapter_config.json']:
            file_from = f"/root/experiments/experiments/{experiment_name}/{filename}"
            file_to   = f"/root/experiments/{experiment_name}/{filename}"
            print(file_from, '->', file_to)
            #!scp -i ~/.ssh/master_hetzner root@65.108.123.219:{file_from} {file_to} 
            os.system(f"scp -i ~/.ssh/master_hetzner root@65.108.123.219:{file_from} {file_to} ")
    else:
        print("File already here")
        
    # experiment_name = "/root/experiments/t2c_concode_220428_v38_plustwoepoch_test_copy/"
    experiment_config = None
    default_model = line['default_model']

    if line['experiment_config.json']:
        experiment_config = json.load(open(f"{EXPERIMENT_PATH}/{experiment_name}/experiment_config.json", 
                                               "r"
                                              )
                                         )

        default_model = json.load(open(f"{EXPERIMENT_PATH}/{experiment_name}/experiment_config.json", 
                                       "r"
                                      )
                                 )['default_model']

    # params_iteration = {"temperature": [1.0],
    #                     "max_new_tokens": [300]#None, 20, 30, 35, 40, 50, 60, 70, 80, 90, 100] + [45, 47, 49, 51, 53, 55]
    #                    }

    print("default_model=", default_model)
    
    
    # Load model
    model = LLaMAForCausalLM.from_pretrained(
        pretrained_model_name_or_path = default_model,
        load_in_8bit=True,
        device_map="auto"
    )

    model = PeftModel.from_pretrained(
            model = model,
            model_id = os.path.join(EXPERIMENT_PATH, experiment_name),
            torch_dtype=torch.float16,
        )

    tokenizer = LLaMATokenizer.from_pretrained(pretrained_model_name_or_path = default_model)

    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    tokenizer.padding_side = "left"
    # not sure how necessary this part is, not sure if tloen/alpaca-lora-7b was even trained with EOS and BOS tokens
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model.eval()
    
    def verbose_model_tokenizer(model, tokenizer):
        print("tokenizer.pad_token_id", tokenizer.pad_token_id)
        print("tokenizer.eos_token_id", tokenizer.eos_token_id)
        print("tokenizer.bos_token_id", tokenizer.bos_token_id)
        print("tokenizer.eos_token_id", tokenizer.eos_token_id)

        print("model.config.pad_token_id", model.config.pad_token_id)
        print("model.config.eos_token_id", model.config.eos_token_id)
        print("model.config.bos_token_id", model.config.bos_token_id)
        print("model.config.eos_token_id", model.config.eos_token_id)
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()

    verbose_model_tokenizer(model, tokenizer)

    # tokenizer.pad_token_id 0
    # tokenizer.eos_token_id 2
    # tokenizer.bos_token_id 1
    # tokenizer.eos_token_id 2
    # model.config.pad_token_id 0
    # model.config.eos_token_id 2
    # model.config.bos_token_id 1
    # model.config.eos_token_id 2
    # trainable params: 0 || all params: 6746804224 || trainable%: 0.0

    generation_config_dict = {"temperature": 1.0,
                              #"penalty_alpha": 2,
                              "max_new_tokens": 500
                             }

    generation_config = GenerationConfig(**generation_config_dict)

    fn_test_data = "/root/data/final_dev.json"
    # fn_test_data = "../data/t2c_answers.json"
    # fn_etalon = "/root/data/answers.json"
    batch_size = 10
    verbose = False

    assert model.training == False

    lst = json.load(open(fn_test_data, 'rb'))
    inputs = lst

    prompts = [generate_test_prompt(inp) for inp in inputs]
    prompts = np.array(prompts)
    print("lst[0]", lst[0])
    print("prompts[0]", prompts[0])
    
    res_list = []
    n = math.ceil(len(prompts)/batch_size)

    for ind in tqdm.tqdm(range(n)):
        current_prompts = prompts[ind*batch_size: (ind+1)*batch_size]
        if verbose:
            print(ind * batch_size, (ind+1)*batch_size, len(current_prompts))

        tokenized_inputs = tokenizer(list(current_prompts), 
                                     padding=True, 
                                     truncation=True, 
                                     return_tensors="pt"
                                    ).to('cuda')



        with torch.no_grad():
            full_output = model.generate(
                **tokenized_inputs,
                generation_config=generation_config
            )

        res_list.extend(tokenizer.batch_decode(full_output, 
                                               skip_special_tokens=False
                                              )
                       )


    res_dict = {"real_config": experiment_config,
                "meta_config": line,
                "generation_config": generation_config_dict,
                "fn_test_data": fn_test_data,
    #             "fn_etalon": fn_etalon,
                "batch_size": batch_size,
                "verbose": verbose,
                "prompts": list(prompts),
                "res_list": res_list
               }

    fn_output = ARTIFACTS_PATH +\
                "predictions" +\
                exp_name+"_"+str(datetime.now()).split('.')[0].replace(' ', '-')
    print(fn_output)

    
    json.dump(res_dict, 
              open(fn_output+'.json', "w+")
             )

    pickle.dump(res_dict,
                open(fn_output+'.pickle', "wb")
               )

    predictions_list = [preprocess(s) for s in res_list]


    fn_results = fn_output+".txt"
    open(fn_results+".txt", 'w+').write('\n'.join(predictions_list))
    # 9321

    
    run_evaluation_of_metrics(fn = fn_results)
    final_metrics = parse_metrics()

    res_dict['final_metrics'] = final_metrics
    json.dump(res_dict, 
          open(fn_output+'_final_metrics.json', "w+")
         )

    pickle.dump(res_dict,
                open(fn_output+'_final_metrics.pickle', "wb")
               )


def main():
    fire.Fire(evaluate)
    
if __name__ == '__main__':
    main()
