import os, sys
import random

from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript

from tools.logger import create_logger
from tools.file_op import write_pkl, write_txt
from tools.gen.gen_tool import load_infos_real_toxicity_prompts, load_infos_dialog_dataset
from tools.search import RedTeamingSearcher
from tools.stat import aggregate_search_result

from config.path import RESULTDIRPATH


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True, 'Evaluate a model')
    parser.add_argument('--synonym_method', type=str, default='bae2')
    parser.add_argument('--attack_method', type=str, default='bayesian')
    parser.add_argument('--exp_name_str', type=str, default='DEBUG')
    parser.add_argument('--user_input_pool_str', type=str, default='zs')
    parser.add_argument('--model_name_str', type=str, default='bb2_3B')
    parser.add_argument('--tc_name_str', type=str, default='default')
    parser.add_argument('--thre', type=float, default=0.0)
    parser.add_argument('--toxicity_obj', type=str, default='question_answer')
    parser.add_argument('--random_perturb', type=bool, default=True)
    parser.add_argument('--tox_range', type=str, default="exp1")
    parser.add_argument('--batch_type', type=str, default="no")
    parser.add_argument('--query_budget', type=int, default=1000)
    parser.add_argument('--expl_budget', type=int, default=50)
    parser.add_argument('--use_sod', type=bool, default=True)
    parser.add_argument('--use_two_GPs', type=bool, default=True)
    parser.add_argument('--use_tox_kernel', type=bool, default=True)

    parser.add_argument('--div_type', type=str, default="none")
    parser.add_argument('--div_coeff', type=float, default=1.0)
    parser.add_argument('--trial_patience', type=int, default=1)
    parser.add_argument('--epsilon', type=int, default=1)
    parser.add_argument('--target_BLEU', type=float, default=0.0)
    parser.add_argument('--num_synonym_candidates', type=int, default=-1)

    parser.set_params(datatype='valid')
    return parser

def get_save_paths(opt):
    EXPPATH = RESULTDIRPATH + f"{opt['exp_name_str']}/" + f"{opt['model_name_str']}_{opt['tc_name_str']}_{opt['user_input_pool_str']}/"
    EXPKEY = f"{opt['synonym_method']}_{opt['attack_method']}_{opt['thre']}_{opt['tox_range']}_{opt['batch_type']}_{opt['query_budget']}_{opt['div_type']}_{opt['div_coeff']}_{opt['trial_patience']}_{opt['epsilon']}_{opt['target_BLEU']}"

    if opt['use_sod'] == True : EXPKEY = EXPKEY + '_sod'
    if opt['use_two_GPs'] == True : EXPKEY = EXPKEY + '_twoGPs'
    if opt['use_tox_kernel'] == True : EXPKEY = EXPKEY + '_utk'
    if opt['toxicity_obj'] == 'question_answer': EXPKEY = EXPKEY + '_qa'
    if opt['num_synonym_candidates'] != -1: EXPKEY = EXPKEY + f"_{opt['num_synonym_candidates']}"
    
    os.makedirs(EXPPATH, exist_ok=True)
    pklpath = EXPPATH + EXPKEY + f'.pkl'
    txtpath = EXPPATH + EXPKEY + f'.txt'
    return pklpath, txtpath

def load_data(opt, printer):
    ## Load Dataset 
    # real-toxicity-prompts : we use offline continuation dataset.
    # ex) user_input_pool_str = 'real-toxicity-prompts/profanity'
    ui_pool, ui_emb, ui_tox, dialog_tox, responses \
            = load_infos_dialog_dataset(dataname=opt['user_input_pool_str'], tox_range=opt['tox_range'], model_name=opt['model_name_str'], toxicity_obj=opt['toxicity_obj'])
    if 'no_precompute' in opt['exp_name_str']:
        dialog_tox, responses = None, None
    return ui_pool, ui_emb, ui_tox, dialog_tox, responses

def global_attack(opt):
    import random
    """
    Get Agent from opt.
    :param opt: tells the agent info
    :return: agent, tasks
    """
    if opt['model_name_str'] == 'bb3_3B_no': opt['rag_retriever_type'] = None
    random.seed(42)

    PKLPATH, TXTPATH = get_save_paths(opt)
    if os.path.exists(PKLPATH.replace('.pkl','_finished.pkl')):
        sys.exit()
    
    logger = create_logger()

    ui_pool, ui_emb, ui_tox, dialog_tox, responses = load_data(opt, printer=logger.info)
    
    logger.info(f"Attack Results for ({opt['synonym_method']}, {opt['attack_method']})")

    searcher = RedTeamingSearcher(
                    opt=opt,
                    ui_pool=ui_pool, 
                    ui_emb=ui_emb, 
                    ui_tox=ui_tox, 
                    dialog_tox=dialog_tox,
                    responses=responses, 
                    pklpath=PKLPATH,
                    printer=logger.info
                )
    results = searcher.run_search()
    stats, results_str = aggregate_search_result(results)
    results['stats'] = stats
    logger.info(results_str)
    os.remove(PKLPATH)
    write_pkl(results, PKLPATH.replace('.pkl','_finished.pkl'))
    write_txt(results_str, TXTPATH.replace('.txt','_finished.txt'))

class GlobalAttack(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        return global_attack(self.opt)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('GlobalAttackerArgument')
    parser.add_argument('--exp_name_str', type=str, default='EXP')
    parser.add_argument('--user_input_pool_str', type=str, default='bot_adversarial_dialogue')
    parser.add_argument('--model_name_str', type=str, default='bb1_3B')
    parser.add_argument('--query_budget', type=int, default=20000)
    # bb1_90M, bb1_3B, bb1_9B, bb2_400M, bb2_3B, bb3_3B

    # Fixed
    parser.add_argument('--tc_name_str', type=str, default='default')
    parser.add_argument('--toxicity_obj', type=str, default='question_answer')
    parser.add_argument('--thre', type=float, default=0.0)
    parser.add_argument('--random_perturb', type=str, default="False")
    parser.add_argument('--batch_type', type=str, default="dpp_posterior") # no, dpp_posterior
    parser.add_argument('--use_sod', type=str, default="True")
    parser.add_argument('--use_two_GPs', type=str, default="True")
    parser.add_argument('--trial_patience', type=int, default=1)
    parser.add_argument('--expl_budget', type=int, default=50)

    parser.add_argument('--synonym_method', type=str, default='bae_nosw')
    parser.add_argument('--attack_method', type=str, default='bayesian')
    parser.add_argument('--tox_range', type=str, default="whole")
    parser.add_argument('--use_tox_kernel', type=str, default="False")
    parser.add_argument('--div_type', type=str, default="bleu_adapt")
    parser.add_argument('--div_coeff', type=float, default=0.3)
    parser.add_argument('--target_BLEU', type=float, default=42.0)
    parser.add_argument('--epsilon', type=int, default=3)
    parser.add_argument('--num_synonym_candidates', type=int, default=20)


    args = parser.parse_args()
    print(' '.join(f'{k}={v}' for k, v in vars(args).items()))
    kwargs = {}
    if args.model_name_str == 'bb1_3B':
        kwargs['model_file'] = 'zoo:blender/blender_3B/model'
 

    # Default inference option is beam search.
    # Since parlai beam search implementation has randomness, 
    # We use Greedy search to induce deterministic behavior.
    if 'bb' in args.model_name_str:
        kwargs['inference'] = 'greedy'
        kwargs['beam_size'] = 1
        
    for key in ['synonym_method','attack_method','exp_name_str','model_name_str','tc_name_str', 'thre', \
        'toxicity_obj','user_input_pool_str','tox_range', 'batch_type', 'query_budget', 'expl_budget', \
            'div_type', 'div_coeff','trial_patience','epsilon', 'target_BLEU', 'num_synonym_candidates']:
        kwargs[key] = getattr(args,key)
    kwargs['random_perturb'] = True if args.random_perturb.lower() == 'true' else False
    kwargs['use_sod'] = True if args.use_sod.lower() == 'true' else False
    kwargs['use_two_GPs'] = True if args.use_two_GPs.lower() == 'true' else False
    kwargs['use_tox_kernel'] = True if args.use_tox_kernel.lower() == 'true' else False
    for key, item in kwargs.items():
        print(key, item)
    results = GlobalAttack.main(
                        datatype='test',
                        **kwargs
                        )
