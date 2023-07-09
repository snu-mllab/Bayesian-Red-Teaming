import torch
import numpy as np
def list_by_indices(l, indices):
    return [l[ind] for ind in indices]

def filter_by_indices(ls, indices):
    filtered_ls = []
    for l in ls:
        if l is None: filtered_ls.append(None)
        elif type(l) == list:
            filtered_ls.append(list_by_indices(l, indices))
        elif type(l) == torch.Tensor or type(l) == np.ndarray:
            filtered_ls.append(l[indices])
        else:
            print(l)
            print(l.size)
            print(type(l))
            raise RuntimeError
    return filtered_ls
