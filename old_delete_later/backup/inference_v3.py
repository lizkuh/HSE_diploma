import torch
from peft import PeftModel
import transformers

# from transformers import LLamaTokenizer, LLamaForCausalLM, GenerationConfig
from transformers import LLaMAForCausalLM, LLaMATokenizer, GenerationConfig

assert torch.cuda.is_available()
import json
import numpy as np
import math
import tqdm
import subprocess



def inference(default_model, 
              experiment_name,
              test_dataset = 't2c_concode/t2c_answers.json',
              fn_answers = "/root/CodeXGLUE/Text-Code/text-to-code/evaluator/answers.json",
              batch_size = 9
             ):
    # load everything
    BASE_MODEL = default_model
    LORA_WEIGHTS = experiment_name #"tloen/alpaca-lora-7b"

    model = LLaMAForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=True,
        device_map="auto"
    )
    print(LORA_WEIGHTS)
    model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        torch_dtype=torch.float16,
    )

    tokenizer = LLaMATokenizer.from_pretrained(BASE_MODEL)
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    tokenizer.padding_side = "left"
    # not sure how necessary this part is, not sure if tloen/alpaca-lora-7b was even trained with EOS and BOS tokens
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model.eval()

    # def generate_prompt(instruction, input):
    #     return f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"

    def generate_test_prompt(data_point, train = False):
        # To decrease expectations of results :)
        assert train == False
        # sorry about the formatting disaster gotta move fast
        if data_point["input"]:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"] if train else ''}"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"] if train else ''}"""


    lst = json.load(open(test_dataset, 'rb'))
    inputs = lst# [lst[0]]
    # instruction = 'Combine the question and answer into an image caption as succinctly as possible. Be sure to include the phrase "a photo of". Do not draw false conclusions.'
    # inputs = ['Is this a baseball game? yes', 'Is this a baseball game? no']
    prompts = [generate_test_prompt(inp) for inp in inputs]
    print(prompts[0])
    prompts = np.array(prompts)
    res_list = []
    for ind in tqdm.tqdm(range(math.ceil(len(prompts)/batch_size))):
        current_prompts = prompts[ind*batch_size: (ind+1)*batch_size]
        print(ind*batch_size, (ind+1)*batch_size, len(current_prompts))

        tokenized_inputs = tokenizer(list(current_prompts), 
                                     padding=True, 
                                     truncation=True, 
                                     return_tensors="pt"
                                    ).to('cuda')

        generation_config = GenerationConfig(max_new_tokens=128)

        with torch.no_grad():
            full_output = model.generate(
                **tokenized_inputs,
                generation_config=generation_config
            )

        res_list.extend(tokenizer.batch_decode(full_output, skip_special_tokens=False))
    #     print(tokenizer.batch_decode(full_output, skip_special_tokens=False)[-1])


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
            "prompts_list": prompts,
            "res_list": res_list,
            "predict_list": predict_list
           }

## example of ussage
# inference(default_model = "decapoda-research/llama-7b-hf", 
#           experiment_name = "t2c_concode_220428_v6"
#          )


