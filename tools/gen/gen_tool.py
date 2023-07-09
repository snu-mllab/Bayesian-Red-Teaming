from tools.file_op import read_pkl
from config.path import DATASETPATH
from tools.eval_utils import get_embedding

import torch

def load_infos_real_toxicity_prompts(key, tox_range):
    path = DATASETPATH + f'real-toxicity-prompts/{key}_{tox_range}_infos.pkl'
    infos = read_pkl(path)
    return infos['prompts'], torch.DoubleTensor(infos['emb']), torch.DoubleTensor(infos['prompts_scores']),\
         torch.DoubleTensor(infos['continuations_scores']), infos['continuations']

def load_infos_dialog_dataset(dataname, tox_range, model_name=None, toxicity_obj='question_answer'):
    path = DATASETPATH + f'{dataname}/{tox_range}_infos.pkl'
    infos = read_pkl(path, print_op=False)

    uip = infos['user_input_pool']
    emb = torch.DoubleTensor(get_embedding(uip))
    tox = torch.DoubleTensor(infos['tox'])
    if f'{model_name}_responses' in infos:
        resps = infos[f'{model_name}_responses']
    else:
        resps = None
    if f'{model_name}_dialog_tox' in infos and toxicity_obj == 'question_answer':
        dtox = torch.DoubleTensor(infos[f'{model_name}_dialog_tox'])
    elif f'{model_name}_response_tox' in infos and toxicity_obj == 'answer':
        dtox = torch.DoubleTensor(infos[f'{model_name}_response_tox'])
    else: 
        dtox = None
    return uip, emb, tox, dtox, resps
