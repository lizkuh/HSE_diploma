class ParamsIterator:
    """
        Recursive get all params all and all
    """
    def __init__(self, params_iteration):
        self.params_iteration = params_iteration
        self.params_list = list(self.params_iteration.keys())
    
    def __len__(self):
        res = 1
        for key in self.params_list:
            res *= len(self.params_iteration[key]) 
        return res
    
    def __iter__(self, 
                 previous_dict = {}
                ):
        if len(self.params_list)==0:
            yield previous_dict
        else:
            param_name = self.params_list.pop()
            for param_val in self.params_iteration[param_name]:
                previous_dict[param_name] = param_val
                yield from self.__iter__(previous_dict=previous_dict)
            self.params_list.append(param_name)
        
# obj = ParamsIterator(params_iteration=params_iteration)
# from tqdm import tqdm
# import time
# for i in tqdm(obj):
#     print(i)
#     time.sleep(1)