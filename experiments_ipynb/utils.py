from EvaluateTestSet import EvaluateTestSet
from transformers import LlamaForCausalLM as LLaMAForCausalLM
from transformers import LlamaTokenizer as LLaMATokenizer

from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model
from peft import PeftModel
import torch

from transformers import GenerationConfig
from tqdm import tqdm_notebook

def load_model_tokenizer_from_pretrained(default_model, 
                                         experiment_name
                                        ):
    # ToDo: BASE model is getting from config
    BASE_MODEL = default_model
    LORA_WEIGHTS = experiment_name

    model = LLaMAForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=True,
        device_map="auto"
    )

    model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        torch_dtype=torch.float16,
    )

    tokenizer = LLaMATokenizer.from_pretrained(BASE_MODEL)
    #model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    #model.config.bos_token_id = 1
    #model.config.eos_token_id = 2

    tokenizer.padding_side = "left"
    # not sure how necessary this part is, not sure if tloen/alpaca-lora-7b was even trained with EOS and BOS tokens
    #raise ValueError("Change this")
    # model.config.bos_token_id = 1
    # model.config.eos_token_id = 2

    model.eval()
    
    return tokenizer, model

import pandas as pd
from matplotlib import pyplot as plt

def draw_metrics_compare_with_glue(data):
    txt = """GrammarT5	Peking University	17/03/2023	24.4	41.54
    StructCoder	Virginia Tech	30/05/2022	22.35	40.91
    JaCoText	Novelis.io	07/12/2021	22.15	39.07
    CoTexT	Case Western Reserve University	23/04/2021	20.1	37.4
    Text2Java-T5	Novelis.io	29/09/2021	21.45	37.46
    PLBART	UCLA & Columbia University	02/04/2021	18.75	36.69
    CodeGPT-adapted	CodeXGLUE Team	30/08/2020	20.1	32.79
    CodeGPT	CodeXGLUE Team	30/08/2020	18.25	28.69
    GPT-2(12L)	CodeXGLUE Team	30/08/2020	17.35	25.37
    Iyer-Simp+200 idoms	CodeXGLUE Team	30/08/2020	12.2	26.6
    Seq2Action+MAML	CodeXGLUE Team	30/08/2020	10.05	24.4
    Seq2Seq	CodeXGLUE Team	30/08/2020	3.05	21.31"""
    
    if isinstance(data, pd.DataFrame):
        data = {'Our': data}
    
    xglue_name_list = [i.split('\t')[0]+f" [{i.split('	')[2]}]" for i in txt.split('\n')]
    xglue_bleu_list = [float(i.split('\t')[4]) for i in txt.split('\n')]



    metric_name = "BLEU"
    param_name = "max_new_tokens"


    plt.figure(figsize = (20, 10))
    plt.title(f"Generation params [{metric_name}]", fontsize = 20)
    
    x_min = 100
    x_max = 0
    for name, _df in data.items():
        df = _df[[param_name, metric_name]].fillna(0).sort_values(param_name)
        print(df.drop_duplicates().shape[0], df[metric_name].nunique(), df[param_name].nunique())
        x_min = min(x_min, df[param_name].min())
        x_max = max(x_max, df[param_name].max())
        plt.plot(df[param_name], df[metric_name], 
                 label = name, 
                 marker = 'x', 
                 markersize = 10
                )

    for i, (name, metric) in enumerate(zip(xglue_name_list, xglue_bleu_list)):
        plt.plot([x_min, x_max], 
                 [metric/100, metric/100], 
                 #label = name + "%2.3f"%(metric/100), 
                 linestyle = '--'
                )
        plt.text(30+(i%3)*10, metric/100, name + " : %2.3f"%(metric/100))

    plt.ylabel(metric_name, fontsize = 16)
    plt.xlabel(param_name, fontsize = 16)
    plt.legend(loc = "upper left", fontsize = 16)
    plt.grid()
    plt.show()
    
def draw_all_metrics(data, param_name = "max_new_tokens"):
    metric_name_list = ['EM', 'BLEU', 'brevity_penalty', 'ratio', 'translation_length',
                   'reference_length', 'precisions_0', 'precisions_1', 'precisions_2',
                   'precisions_3']
    
    for metric_name in metric_name_list:
        if metric_name == param_name:
            continue

        plt.figure(figsize = (20, 10))
        plt.title(f"Generation params [{metric_name}]", fontsize = 20)

        df = data[[param_name, metric_name]].fillna(0).sort_values(param_name)
        print(df.drop_duplicates().shape[0], df[metric_name].nunique(), df[param_name].nunique())
        plt.plot(df[param_name], df[metric_name], 
                 label = "Our", 
                 marker = 'x', 
                 markersize = 10
                )

        plt.ylabel(metric_name, fontsize = 16)
        plt.xlabel(param_name, fontsize = 16)
        plt.legend(loc = "upper left", fontsize = 16)
        plt.grid()
        plt.show()