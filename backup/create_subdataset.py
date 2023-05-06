import os
import json
import pandas as pd
from sklearn.utils import shuffle

def show_datasets_size(input_path = "alpaca-lora/t2c_concode"):
    dct = {}
    for filename in os.listdir(input_path):
        if filename.endswith('.json'):
            dct[filename] = json.load(open(f"{input_path}/{filename}", "r"))

    df = {}
    keys = list(dct.keys())
    for key1 in keys:
        df[key1] = {}
        for key2 in keys:
            lst1 = [i['output'] for i in dct[key2]]
            lst2 = [i['output'] for i in dct[key1]]

            df[key1][key2] = len(set(lst1)&set(lst2))

    print(pd.DataFrame(df).to_markdown())
    return df

def create_subsable_dataset(fn_train, 
                            fn_val, 
                            fn_output, 
                            n_samples = 10001
                           ):
    lst = set([i['output'] for i in json.load(open(fn_val, 'r'))])
    lst_new = [i for i in json.load(open(fn_train, 'r')) if i['output'] not in lst]
    lst_new = shuffle(lst_new)[:n_samples]
    json.dump(lst_new, open(fn_output, 
                            "w+"
                           )
             )
    
    return
                                    