import torch
from transformers import GenerationConfig

from datetime import datetime
import json
import numpy as np
import math
import tqdm

from T2CEvaluator import T2CEvaluator

from prompter import Prompter
prompter = Prompter()

def generate_test_prompt(data_point):
    #assert 'output' not in data_point or data_point['output']==''
    if "input" in data_point and data_point["input"]:
        return prompter.generate_prompt(instruction = data_point["instruction"],
                                        input = data_point["input"],
                                        #label = ''#data_point["output"]
                                       )
    else:
        return prompter.generate_prompt(instruction = data_point["instruction"],
                                        #input = None,
                                        #label = ''#data_point["output"]
                                       )
   
# def generate_test_prompt(data_point, train = False):
#     # To decrease expectations of results :)
#     assert train == False
#     # sorry about the formatting disaster gotta move fast
#     if data_point["input"]:
#         return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
# ### Instruction:
# {data_point["instruction"]}

# ### Input:
# {data_point["input"]}

# ### Response:
# {data_point["output"] if train else ''}"""
#     else:
#         return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
# ### Instruction:
# {data_point["instruction"]}

# ### Response:
# {data_point["output"] if train else ''}"""


class EvaluateTestSet:
    def __init__(self, 
                 generation_config = GenerationConfig(max_new_tokens = 128), 
                 fn_test_data = "../data/t2c_answers.json",
                 fn_etalon = "/root/data/answers.json",
                 batch_size = 10,
                 verbose = False
                ):
        self.generation_config = generation_config
        
        self.fn_test_data = fn_test_data
        self.fn_etalon = fn_etalon
        
        self.batch_size = batch_size
        self.verbose = verbose
       
    def preprocess(self, s):
        #ToDo rewrite it using Promt Template
        #prompter.get_response(s)
        s = s.split('### Response:\n')[-1]
        s = s.replace('\n', '  ')
        s = s.replace('<unk>', " ")
        s = ' '.join(s.split(' ')[:100])
        while '  ' in s:
            s = s.replace('  ', ' ')

        if len(s) > 0 and s[0] == ' ':
            s = s[1:]
        
        if self.verbose:
            print(s)
        
        return s

    def clean_results(self, res_list):
        predict_list = []
        for s in tqdm.tqdm(res_list):
            predict_list.append(self.preprocess(s))
        return predict_list
    
    def get_raw_results(self, model, tokenizer, prompts):
        batch_size = self.batch_size
        generation_config = self.generation_config
        
        res_list = []
        n = math.ceil(len(prompts)/batch_size)
        
        for ind in tqdm.tqdm(range(n)):
            current_prompts = prompts[ind*batch_size: (ind+1)*batch_size]
            if self.verbose:
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

            res_list.extend(tokenizer.batch_decode(full_output, skip_special_tokens=False))
        
        return res_list
    
    def save_results(self, predict_list):
        output_filename = str(datetime.now()).split('.')[0].replace(' ', '-').replace(':', '_')+'.txt'
        fn_output = "/root/results/%s"%output_filename
        
        res = '\n'.join([i if i!='' else '-' for i in predict_list])
        open(fn_output, "w+", encoding='utf-8').write(res)
        return fn_output
    
    def evaluate(self, model, tokenizer):
        model.eval()
        assert model.training == False

        lst = json.load(open(self.fn_test_data, 'rb'))
        inputs = lst# [lst[0]]
        # instruction = 'Combine the question and answer into an image caption as succinctly as possible. Be sure to include the phrase "a photo of". Do not draw false conclusions.'
        # inputs = ['Is this a baseball game? yes', 'Is this a baseball game? no']
        prompts = [generate_test_prompt(inp) for inp in inputs]
        prompts = np.array(prompts)
        
        res_list = self.get_raw_results(model = model, 
                                        tokenizer = tokenizer,
                                        prompts = prompts)
        
        model.train()
        assert model.training == True
        
        predict_list = self.clean_results(res_list)
        
        self.fn_output = self.save_results(predict_list)
        
        t2c_evaluator = T2CEvaluator()
        metric_res = t2c_evaluator.calculate_metrics(fn_answers = self.fn_etalon, 
                                                     fn_predictions = self.fn_output
                                                     )
        return metric_res