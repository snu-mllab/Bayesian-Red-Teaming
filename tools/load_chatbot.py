from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from tools.toxicity import BlackBoxToxicity
from tools.dialogpt_toxicity import DialoGPTBlackBoxToxicity
from tools.godel_toxicity import GODELBlackBoxToxicity
from parlai.core.script import ParlaiScript
from parlai.core.params import ParlaiParser
import os
def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True, 'Evaluate a model')
        parser.add_argument('--model_name_str', type=str, default='bb1_90M')
        parser.set_params(datatype='test')
        return parser

def load_chatbot(opt, printer=print):
    if 'bb' in opt['model_name_str'] or 'seeker' in opt['model_name_str']:
        bot_agent = create_agent(opt, requireModelExists=True)
    else:
        bot_agent = opt['model_name_str']
    for key, item in opt.items():
        if 'beam' in key or 'inference' in key:
            print(key, item)
    return bot_agent  

def load_chatbot_and_bbf(opt, printer=print):
    if 'bb' in opt['model_name_str'] or 'seeker' in opt['model_name_str']:
        bot_agent = create_agent(opt, requireModelExists=True)
        black_box_func = BlackBoxToxicity(
                                        bot_agent=bot_agent, 
                                        toxicity_obj=opt['toxicity_obj'], 
                                        tc_name_str=opt['tc_name_str'], 
                                        random_perturb=opt['random_perturb'],
                                        synonym_method=opt['synonym_method'],
                                        printer=printer
                                        )
    elif opt['model_name_str'] in ['DialoGPT-large','DialoGPT-medium','DialoGPT-small']:
        bot_agent = opt['model_name_str']
        black_box_func = DialoGPTBlackBoxToxicity(bot_agent=bot_agent, toxicity_obj=opt['toxicity_obj'], tc_name_str=opt['tc_name_str'], random_perturb=opt['random_perturb'], synonym_method=opt['synonym_method'],printer=printer)
    elif opt['model_name_str'] in ['GODEL-base','GODEL-large']:
        bot_agent = opt['model_name_str']
        black_box_func = GODELBlackBoxToxicity(bot_agent=bot_agent, toxicity_obj=opt['toxicity_obj'], tc_name_str=opt['tc_name_str'], random_perturb=opt['random_perturb'], synonym_method=opt['synonym_method'],printer=printer)
    return bot_agent, black_box_func

class Loader(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        return load_chatbot(self.opt)

def load_chatbot_by_name(model_name_str, use_greedy=True):
    kwargs = {'model_name_str': model_name_str}
    if model_name_str == 'bb1_3B':
        kwargs['model_file'] = 'zoo:blender/blender_3B/model'
    
    if use_greedy:
        # Default inference option is beam search.
        # Since parlai beam search implementation has randomness, 
        # We use Greedy search to 
        kwargs['inference'] = 'greedy'
        kwargs['beam_size'] = 1
    bot_agent = Loader.main(
                        datatype='test', 
                        **kwargs
                        )

    return bot_agent

if __name__ == '__main__':
    bot_agent = load_chatbot_by_name('bb2_3B')
    print(bot_agent)
