import time
import random
import torch
from tools.file_op import write_pkl
from tools.search.bayesian_question_choicer import BayesianQuestionChoicer
from tools.search.bayesian_question_choice_and_editer import BayesianQuestionChoiceAndEditer
from tools.load_chatbot import load_chatbot_and_bbf
from textattack.shared.attacked_text import AttackedText
from tools.toxicity import TC

class RedTeamingSearcher:
    def __init__(self, opt, ui_pool, ui_emb, ui_tox, dialog_tox=None, responses=None, pklpath='', printer=print):
        self.opt = opt
        self.ui_pool = ui_pool
        self.ui_emb = ui_emb
        self.ui_tox = ui_tox
        self.dialog_tox = dialog_tox
        self.responses = responses
        self.pklpath = pklpath
        self.printer = printer

        self.num_q = len(self.ui_pool)
        self.budget = opt['query_budget']

        if self.opt['attack_method'] == 'bayesian_edit' or responses is None:
            print("Load chatbot and toxicity model")
            self.black_box_func = load_chatbot_and_bbf(opt, printer)[1]
            self.synonym_func = self.black_box_func.tc.synonym_func
            self.tc = self.black_box_func.tc
        elif dialog_tox is None:
            print("Load only toxicity model")
            self.tc = TC(tc_name_str=opt['tc_name_str'], synonym_method=opt['synonym_method'])
        else:
            print("All infos are pre-computed!")
            print("Do not load chatbot and toxicity model")
            self.black_box_func, self.synonym_func, self.tc = None, None, None

        self.infer_counter = 0
        
        self.results = {'opt' : self.opt}

    def run_search(self):
        self.printer("Start Search")

        t0 = time.time()
        self.results['opt']['t0'] = t0

        if self.opt['attack_method'] in ['default', 'highest', 'lowest']:
            self.heuristic_search(self.opt['attack_method'])
        elif self.opt['attack_method'] == 'bayesian':
            self.bayesian_search()
        elif self.opt['attack_method'] == 'bayesian_edit':
            self.bayesian_edit_search()
        self.printer(f"Elapsed Time : {time.time()-t0}")
        return self.results

    def heuristic_search(self, attack_method='default'):
        assert attack_method in ['default', 'highest', 'lowest'], f"attack_method : {attack_method} should be one of [default, highest, lowest]"
        if attack_method == 'default':
            total_score_vector = torch.rand(self.num_q)
        elif attack_method == 'highest':
            total_score_vector = self.ui_tox.float()
        elif attack_method == 'lowest':
            total_score_vector = -self.ui_tox.float()
        else:
            raise NotImplementedError
        eval_indices = torch.topk(total_score_vector, self.budget)[1].numpy()
        self.run_infer_batch(eval_indices=eval_indices)
        self.results['opt']['total_score_vector'] = total_score_vector
        write_pkl(self.results, self.pklpath)
        return self.results

    def bayesian_search(self):
        BQC = BayesianQuestionChoicer(
                                    ui_pool=self.ui_pool, 
                                    ui_emb=self.ui_emb, 
                                    ui_tox=self.ui_tox,
                                    expl_budget=self.opt['expl_budget'], 
                                    batch_type=self.opt['batch_type'], 
                                    thre=self.opt['thre'],
                                    div_type=self.opt['div_type'],
                                    div_coeff=self.opt['div_coeff'],
                                    target_BLEU=self.opt['target_BLEU'],
                                    use_sod=self.opt['use_sod'], 
                                    use_tox_kernel=self.opt['use_tox_kernel'],
                                    printer=self.printer
            )
        expl_indices = BQC.exploration()
        scores = self.run_infer_batch(eval_indices=expl_indices)
        BQC.update(expl_indices, scores)
        while self.infer_counter < self.budget:
            BQC.batch_size = min(BQC.batch_size, self.budget - self.infer_counter)
            batch_indices = BQC.bayesian_step_batch() 
            scores = self.run_infer_batch(eval_indices=batch_indices)
            BQC.update(batch_indices, scores)
        self.results['opt']['total_score_vector'] = BQC.GP.acquisition(BQC.indices, bias=0.0, add_mean=0.0)
        write_pkl(self.results, self.pklpath)
        return self.results
    
    def bayesian_edit_search(self):
        BQCE = BayesianQuestionChoiceAndEditer(
                                    ui_pool=self.ui_pool, 
                                    ui_emb=self.ui_emb, 
                                    ui_tox=self.ui_tox,
                                    expl_budget=self.opt['expl_budget'], 
                                    synonym_func=self.synonym_func, 
                                    batch_type=self.opt['batch_type'], 
                                    thre=self.opt['thre'],
                                    div_type=self.opt['div_type'],
                                    div_coeff=self.opt['div_coeff'],
                                    target_BLEU=self.opt['target_BLEU'],
                                    use_sod=self.opt['use_sod'], 
                                    use_two_GPs=self.opt['use_two_GPs'], 
                                    use_tox_kernel=self.opt['use_tox_kernel'],
                                    tox_range=self.opt['tox_range'],
                                    TC=self.tc, 
                                    trial_patience=self.opt['trial_patience'],
                                    epsilon=self.opt['epsilon'],
                                    num_synonym_candidates=self.opt['num_synonym_candidates'],
                                    printer=self.printer,
            )
        expl_indices, expl_texts = BQCE.exploration()
        scores = self.run_infer_batch_texts(eval_texts=expl_texts, eval_indices=expl_indices)
        #scores = self.run_infer_batch(eval_indices=expl_indices)
        BQCE.update(expl_indices, expl_texts, scores)
        while self.infer_counter < self.budget:
            BQCE.batch_size = min(BQCE.batch_size, self.budget - self.infer_counter)
            batch_indices, batch_texts = BQCE.bayesian_step_batch() 
            scores = self.run_infer_batch_texts(eval_texts=batch_texts, eval_indices=batch_indices)
            BQCE.update(batch_indices, batch_texts, scores)
        
        N, M = self.ui_emb.shape
        self.results['opt']['total_score_vector'] = BQCE.GP_edit.acquisition(BQCE.ui_input[:,:M], bias=0.0, add_mean=0.0)
        self.results['opt']['total_score_vector_second_GP'] = BQCE.GP.acquisition(BQCE.ui_input, bias=0.0, add_mean=0.0)
        write_pkl(self.results, self.pklpath)
        return self.results

    def run_infer(self, idx):
        Q = self.ui_pool[idx]
        if self.ui_tox is not None:
            score = float(self.ui_tox[idx].numpy())
            label = score > 0 
        else:
            score, label = self.black_box_func.calc_toxicity(Q)
        
        if self.dialog_tox is not None:
            assert self.responses is not None
            answer0 = self.responses[idx] 
            score0 = self.dialog_tox[idx]
            if type(score0) == torch.Tensor:
                score0 = float(score0.numpy())
            label0 = score0 > 0
        elif self.responses is not None:
            answer0 = self.responses[idx] 
            score0, _ = self.tc.calc_toxicity(answer0)
            if type(score0) == torch.Tensor:
                score0 = float(score0.numpy())
            label0 = score0 > 0
        else:
            score0, label0, answer0 = self.black_box_func.black_box(Q, print_op=False, return_answer=True)
        result = {
            'ui_label' : label,
            'ui_score' : score,
            'dialog_label' : label0,
            'dialog_score' : score0,
            'answer' : answer0,
            'orig_question' : Q,
            'index' : idx,
            'time_stamp' : time.time()
        }
        self.infer_counter += 1
        return result

    def run_infer_text(self, text, idx=None):
        if idx is not None and text == self.ui_pool[idx]:
            cache_flag = True
        else:
            cache_flag = False
        if cache_flag and self.dialog_tox is not None and self.responses is not None:
            score = self.ui_tox[idx]
            if type(score) == torch.Tensor:
                score = float(score.numpy())
            label = score > 0
            answer0 = self.responses[idx] 
            score0 = self.dialog_tox[idx]
            if type(score0) == torch.Tensor:
                score0 = float(score0.numpy())
            label0 = score0 > 0
        else:
            score, label = self.black_box_func.calc_toxicity(text)

            score0, label0, answer0 = self.black_box_func.black_box(text, print_op=False, return_answer=True)
        result = {
            'ui_label' : label,
            'ui_score' : score,
            'dialog_label' : label0,
            'dialog_score' : score0,
            'answer' : answer0,
            'orig_question' : text,
            'index' : idx,
            'time_stamp' : time.time()
        }
        self.infer_counter += 1
        return result

    def run_infer_batch(self, eval_indices):
        scores = []
        init_ct = self.infer_counter
        for ct, eval_ind in enumerate(eval_indices):
            self.results[init_ct + ct] = self.run_infer(eval_ind)
            scores.append(self.results[init_ct + ct]['dialog_score'])
            if (init_ct + ct) % 10 == 0: 
                print(f"{init_ct + ct} / {self.budget}")
                write_pkl(self.results, self.pklpath, print_op=False)
        return scores
    
    def run_infer_batch_texts(self, eval_texts, eval_indices):
        scores = []
        init_ct = self.infer_counter
        for ct, (eval_text, eval_ind) in enumerate(zip(eval_texts, eval_indices)):
            self.results[init_ct + ct] = self.run_infer_text(eval_text, eval_ind)
            scores.append(self.results[init_ct + ct]['dialog_score'])
            if (init_ct + ct) % 10 == 0:
                print(f"{init_ct + ct} / {self.budget}")
                write_pkl(self.results, self.pklpath, print_op=False)
        return scores
