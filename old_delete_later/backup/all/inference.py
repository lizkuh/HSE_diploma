import os
import sys

import fire
# import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LLaMAForCausalLM, LLaMATokenizer

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

import json
import tqdm
import math
import subprocess

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

# python inference.py --fn_dataset='t2c_concode/t2c_10answers.json' --fn_answers="t2c_concode/etalon/10answers.jsonl" --base_model="decapoda-research/llama-7b-hf" --lora_weights="t2c_concode_220428_finetune_v7" --batch_size=None

# default_model = "decapoda-research/llama-7b-hf", 
#                          experiment_name = "alpaca-lora/t2c_concode_220428_v6",
#                          test_dataset = 'alpaca-lora/t2c_concode/t2c_10answers.json',
#                          fn_answers = "alpaca-lora/t2c_concode/etalon/10answers.jsonl",
#                          batch_size = 10


def main(
    fn_dataset: str,
    fn_answers: str,
    base_model: str,# = "",
    lora_weights: str,# = "tloen/alpaca-lora-7b",
    load_8bit: bool = False,
    batch_size = 10,
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
#     server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
#     share_gradio: bool = False,
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LLaMATokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LLaMAForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = LLaMAForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LLaMAForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    
    ## comment this line?
    #if torch.__version__ >= "2" and sys.platform != "win32":
    #    print("torch compile")
    #    model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        temperature=1,#0.1,
        top_p=1,#0.75,
        top_k=50,
        num_beams=1,#,
        min_new_tokens=64,#new
        max_new_tokens=512,
        #stream_output=False,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }
        
        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return prompter.get_response(output)

    inputs = json.load(open(fn_dataset, 'rb'))
    res_list = []
    for input in tqdm.tqdm(inputs):
        current_res = evaluate(input['instruction'], 
                               input['input']
                              )
        res_list.append(current_res)
#     for ind in tqdm.tqdm(range(math.ceil(len(inputs)/batch_size))):
#         current_prompts = inputs[ind*batch_size: (ind+1)*batch_size]
#         current_res = evaluate(current_prompts)
# #         print(ind*batch_size, (ind+1)*batch_size, len(current_prompts))

# #         tokenized_inputs = tokenizer(list(current_prompts), 
# #                                      padding=True, 
# #                                      truncation=True, 
# #                                      return_tensors="pt"
# #                                     ).to('cuda')

# #         generation_config = GenerationConfig(max_new_tokens=128)

# #         with torch.no_grad():
# #             full_output = model.generate(
# #                 **tokenized_inputs,
# #                 generation_config=generation_config
# #             )

#         res_list.extend(current_res)#tokenizer.batch_decode(full_output, skip_special_tokens=False))
#     #     print(tokenizer.batch_decode(full_output, skip_special_tokens=False)[-1])


        def preprocess(s, verbose = False):
            s = s.split('### Response:\n')[-1]
            s = s.replace('\n', '  ')
            s = s.replace('<unk>', " ")
            s = ' '.join(s.split(' ')[:100])
            while '  ' in s:
                s = s.replace('  ', ' ')

            if len(s) > 0 and s[0] == ' ':
                s = s[1:]
            
            if verbose:
                print(s)
            
            if s == '':
                print("empty res")
                return ' '
            
            return s


    predict_list = []
    for s in tqdm.tqdm(res_list):
        predict_list.append(preprocess(s))

    from datetime import datetime
    output_filename = str(datetime.now()).split('.')[0].replace(' ', '-').replace(':', '_')+'.txt'
    current_pred_name = "/root/alpaca-lora/t2c_concode/results/%s"%output_filename
    print("Save results at file", current_pred_name)

    with open(current_pred_name, 'w+') as f:
        f.write('\n'.join(predict_list))
    #     f.write('\n')

    command = f"python /root/CodeXGLUE/Text-Code/text-to-code/evaluator/evaluator.py -a {fn_answers} -p {current_pred_name}"
    result = subprocess.run(command.split(' '), 
                            stderr=subprocess.PIPE)
    result.stderr
    results = result.stderr.decode()
    print(results)
    
    return {"metric_string": results,
            "inputs": inputs,
            #"prompts_list": prompts,
            "res_list": res_list,
            "predict_list": predict_list
           }

## example of ussage
# inference(default_model = "decapoda-research/llama-7b-hf", 
#           experiment_name = "t2c_concode_220428_v6"
#          )




if __name__ == "__main__":
    fire.Fire(main)
