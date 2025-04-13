import torch
from torch.nn.utils.rnn import pad_sequence


def convert_to_fit_npz(obj):
    '''
    convert nested list to array: homogeneous 
    '''
    if isinstance(obj[0][0], list):
        # if 3th nested list, convert it to 2rd nested list firstly
        obj_lst = []
        for nest_lst in obj:
            nonest_lst = []
            for lst in nest_lst:
                nonest_lst = nonest_lst + lst
            obj_lst.append(nonest_lst)
    
        obj = obj_lst
        
    return_obj = pad_sequence([torch.tensor(r,dtype=torch.int64) 
                                      for r in obj], 
                                     batch_first=True, padding_value=0)
    return return_obj