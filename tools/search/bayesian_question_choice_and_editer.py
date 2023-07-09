import random
import numpy as np
import gc, torch
from tools.toxicity import get_tox_range
from tools.search.bayesopt.surrogate_model.gp_model_base import MyGPModelBase
from tools.search.bayesopt.dpp.dpp import dpp_sample
from tools.search.bayesopt.acquisition.algorithm.kmeanspp import kmeans_pp_cuda_cosine
from tools.slice_op import list_by_indices
from tools.eval_utils import get_embedding
from tools.bleu import BLEULauncher, SBERTLauncher, get_self_bleu
from collections import Counter
import time
class DataInfo:
    def __init__(self):
        self.indices = []
        self.scores = []
        self.texts = []
        self.embs = torch.empty(0).double()
        self.embs_input = torch.empty(0).double()

    def add_datum(self, ind, score, text, emb, emb_input):
        self.indices.append(ind)
        self.scores.append(score)
        self.texts.append(text)
        self.embs = torch.cat([self.embs, emb], axis=0)
        self.embs_input = torch.cat([self.embs_input, emb_input], axis=0)

    def add_data(self, indices, scores, texts, embs, embs_input):
        self.indices.extend(indices)
        self.scores.extend(scores)
        self.texts.extend(texts)
        self.embs = torch.cat([self.embs, embs], axis=0)
        self.embs_input = torch.cat([self.embs_input, embs_input], axis=0)

    def set_data(self, indices, scores, texts, embs, embs_input):
        self.indices = indices
        self.scores = scores
        self.texts = texts
        self.embs = embs
        self.embs_input = embs_input
    
    def __getitem__(self, idx):
        return (self.indices[idx], self.scores[idx], self.texts[idx], self.embs[idx].view(1,-1), self.embs_input[idx].view(1,-1))
    
    def get_item(self, idx):
        return self.indices[idx], self.scores[idx], self.texts[idx], self.embs[idx].view(1,-1), self.embs_input[idx].view(1,-1)
    
    def get_items(self, indices):
        return list_by_indices(self.indices, indices),\
                list_by_indices(self.scores, indices),\
                list_by_indices(self.texts, indices),\
                self.embs[indices],\
                self.embs_input[indices]
    def get_all_items(self):
        return self.get_items(list(range(len(self.indices))))


class BayesianQuestionChoiceAndEditer:
    def __init__(
            self, 
            ui_pool, 
            ui_emb, 
            ui_tox, 
            expl_budget, 
            synonym_func=None, 
            batch_type='dpp_posterior', 
            thre=0.0, 
            div_type='none',
            div_coeff=0.0,
            target_BLEU=None,
            use_sod=True, 
            use_two_GPs=True, 
            use_tox_kernel=False,
            tox_range='whole', 
            TC=None, 
            trial_patience=1,
            epsilon=1,
            num_synonym_candidates=-1,
            printer=print
        ):
        self.printer = printer
        self.counter = 0

        self.ui_pool = ui_pool
        self.ui_emb = ui_emb
        self.ui_tox = ui_tox
        self.expl_budget = expl_budget
        self.synonym_func = synonym_func
        self.num_synonym_candidates = num_synonym_candidates
        self.indices = list(range(len(self.ui_pool)))
        self.eval_DI = DataInfo()
        self.fit_DI = DataInfo()
        self.succ_DI = DataInfo()
        self.use_tox_kernel = use_tox_kernel


        # ui_input is user input's feature for GP. (this GP estimate after-edit score)
        self.ui_input = self.ui_emb.clone()
        if self.use_tox_kernel:
            # if we have toxicity of user inputs, we concat toxicity score to embedding to get better feature
            assert ui_tox is not None
            self.ui_tox = torch.DoubleTensor(self.ui_tox)
            self.ui_input = torch.cat([self.ui_input,self.ui_tox.view(-1,1)], axis=1)


        self.GP = MyGPModelBase(fit_iter=20) 
        self.GP_edit = MyGPModelBase(fit_iter=20) 
        self.batch_size = 10
        self.batch_type = batch_type
        self.thre = thre
        self.use_sod = use_sod
        self.sod_size = 1000
        self.sod_limit = 10000

        self.TC = TC
        self.tox_min, self.tox_max = get_tox_range(tox_range)
        self.use_two_GPs = use_two_GPs
        self.memory_count = 0

        self.trial_counter = Counter()
        self.stop_indices = []
        self.trial_patience = trial_patience
        self.epsilon = epsilon

        self.bias = -1.0

        assert div_type in ['none','bleu','sbert','bleu_adapt'], f"div type should be one of [none, bleu, sbert, bleu_adapt] not {div_type}"
        self.div_type = div_type
        self.div_coeff = div_coeff
        self.target_BLEU = target_BLEU
        self.init_div()


    def clean(self):
        self.eval_DI = DataInfo()
        self.fit_DI = DataInfo()
        self.succ_DI = DataInfo()

        self.trial_counter = Counter()
        self.stop_indices = []

        self.GP = MyGPModelBase(fit_iter=20) 
        self.GP_edit = MyGPModelBase(fit_iter=20) 
        self.batch_size = 10
        self.memory_count = 0
        self.init_div()

    def init_div(self):
        self.div_vector = torch.zeros(len(self.ui_emb)).cuda()
        if self.div_type == 'bleu' or self.div_type == 'bleu_adapt':
            self.div_launcher = BLEULauncher()
            self.div_launcher.add_hyps(self.ui_pool)
            if self.div_type == 'bleu_adapt':
                self.printer(f"opt['div_coeff']={self.div_coeff} is used as initial div_coeff of bleu_adapt method.")
                self.adapt_min = 0.0001
                self.adapt_max = 100.0
                self.adapt_step_size = 1.01
                self.adapt_gap = 1.0
                if self.div_coeff > self.adapt_max or self.adapt_min > self.div_coeff:
                    self.printer(f"initial div coeff should be between {self.adapt_min} and {self.adapt_max}")
                    raise RuntimeError
                if self.target_BLEU is None:
                    self.printer("You should set opt['target_BLEU'].")
                    raise RuntimeError
        elif self.div_type == 'sbert':
            self.div_launcher = SBERTLauncher()
            self.div_launcher.add_hyps(self.ui_pool, self.ui_emb)
        self.is_div_launcher_available = False

    def subset_of_data(self, embs):
        assert len(embs) > self.sod_size, "something wrong"
        _, selected_indices = kmeans_pp_cuda_cosine(
                                        data = embs, 
                                        k = self.sod_size
                                    )
        return selected_indices
    
    def subsample_fit_indices_by_sod(self):
        if len(self.eval_embs) > self.sod_limit:
            rnd_inds = random.sample(range(len(self.eval_embs)), self.sod_limit)
            self.fit_DI.set_data(*self.eval_DI.get_items(rnd_inds))
        else:
            self.fit_DI.set_data(*self.eval_DI.get_all_items())
        selected_indices = self.subset_of_data(self.fit_embs)
        self.fit_DI.set_data(*self.fit_DI.get_items(selected_indices))

    def exploration(self):
        sampled_indices = random.sample(self.indices, self.expl_budget)
        return sampled_indices, list_by_indices(self.ui_pool, sampled_indices)

    def update(self, indices, texts, scores):
        self.clean_memory_cache()
        embs = get_embedding(texts)
        embs_input = embs.clone()
        
        # update eval, fit data
        data = {
            'indices': indices,
            'texts': texts,
            'scores': scores,
            'embs': embs,
            'embs_input': embs_input,
        }
        self.fit_DI.add_data(**data)
        self.eval_DI.add_data(**data)
        
        # update success data
        for ind, score, text, emb, emb_input in zip(indices, scores, texts, embs, embs_input):
            if score > self.thre:
                self.succ_DI.add_datum(ind, score, text, emb.view(1,-1), emb_input.view(1,-1))
                if self.div_type == 'sbert':
                    self.div_launcher.add_refs([text], emb.view(1,-1))

        # use sod to get smaller fit data
        if self.use_sod and len(self.eval_indices) > self.sod_size:
            self.subsample_fit_indices_by_sod()
        
        # fit GP model
        self.GP_edit.fit(self.fit_embs_input, self.fit_scores)
        if self.use_two_GPs:
            self.GP.fit(self.ui_input[self.fit_indices], self.fit_scores)
        else:
            self.GP = self.GP_edit

        # update diversity term
        if self.counter % 10 == 0 and len(self.succ_indices) > 100:
            self.is_div_launcher_available = True
            indices = list(range(len(self.succ_indices)))
            cur_ref_indices = random.sample(indices, 500) if len(indices) >= 500 else indices
            if self.div_type == 'bleu' or self.div_type == 'bleu_adapt':
                self.div_launcher.init_refs(list_by_indices(self.succ_texts, cur_ref_indices))
                self.div_vector = torch.DoubleTensor(self.div_launcher.calc_bleu_all()).cuda()
            elif self.div_type == 'sbert':
                assert len(self.succ_indices) == len(self.div_launcher.refs)
                self.div_launcher.set_cur_ref_indices(cur_ref_indices)
                self.div_vector = torch.DoubleTensor(self.div_launcher.calc_sbert_all_by_ref_indices()).cuda()
            assert self.div_vector.shape == (len(self.ui_pool),)

        # update div_coeff if div_type == bleu_adapt
        if self.div_type == 'bleu_adapt':
            self.adapt_div_coeff()

        # update bias
        self.bias = torch.DoubleTensor(self.eval_scores).clone().cuda()
        self.bias[self.bias > 0] = 0.0 
        if self.is_div_launcher_available:
            if self.div_type == 'sbert':
                self.bias += self.div_coeff * torch.DoubleTensor(self.div_launcher.calc_sbert_by_ref_indices(self.eval_embs)).cuda()
            elif self.div_type == 'bleu' or self.div_type == 'bleu_adapt':
                self.bias += self.div_coeff * torch.DoubleTensor(self.div_launcher.calc_bleus(self.eval_texts)).cuda()
        self.bias = torch.max(self.bias)
        self.counter += 1

    def adapt_div_coeff(self):
        if self.is_div_launcher_available:
            # calc average self bleu
            cur_self_bleus = []
            for i in range(10):
                sampled_texts = random.sample(self.succ_texts, 100)
                cur_self_bleus.append(get_self_bleu(sampled_texts))
            cur_self_bleu = np.mean(cur_self_bleus)
            if cur_self_bleu > self.target_BLEU:
                self.div_coeff = min(self.adapt_max, self.div_coeff * self.adapt_step_size)
            elif cur_self_bleu < self.target_BLEU - self.adapt_gap:
                self.div_coeff = max(self.adapt_min, self.div_coeff / self.adapt_step_size)

    def filter_candidates(self, texts, filter_refs=[]):
        if len(texts)==0: return texts
        if self.tox_min==-1.0 and self.tox_max==1.0:
            new_texts = []
            for text in texts:
                if text not in self.eval_texts and text not in filter_refs:
                    new_texts.append(text)
        else:
            scores, _ = self.TC.calc_toxicity_batch(texts)
            new_texts = []
            for text, score in zip(texts, scores):
                # edited text should satisfy the toxicicty condition.
                if score >= self.tox_min and score <= self.tox_max:
                    if text not in self.eval_texts and text not in filter_refs:
                        new_texts.append(text)
        return new_texts

    def calc_acquisitions_by_indices(self, candidates):
        '''
            candidates : indices (List[int])
        '''
        if self.div_type == 'none':
            add_mean = 0.0
        else:
            add_mean = self.div_coeff * self.div_vector[candidates]
        cand_input = self.ui_input[candidates]
        acquisitions = self.GP.acquisition(cand_input, bias=self.bias, add_mean=add_mean)
        del add_mean, cand_input
        return acquisitions
    
    def calc_acquisitions_by_emb(self, text_candidates):
        '''
            text_candidates : texts (List[str])
        '''
        candid_emb = get_embedding(text_candidates)
        candid_emb_input = candid_emb.clone()
        
        # if self.use_tox_kernel:
        #     candid_emb_tox, _ = self.TC.calc_toxicity_batch(text_candidates)
        #     candid_emb_tox = torch.DoubleTensor(candid_emb_tox)
        #     candid_emb_input = torch.cat([candid_emb_input, candid_emb_tox.view(-1,1)], axis=1)

        if self.div_type == 'none' or not self.is_div_launcher_available:
            add_mean = 0.0
        elif self.div_type == 'bleu' or self.div_type == 'bleu_adapt':
            add_mean = self.div_coeff * torch.DoubleTensor(self.div_launcher.calc_bleus(text_candidates)).cuda()
        elif self.div_type == 'sbert':
            add_mean = self.div_coeff * torch.DoubleTensor(self.div_launcher.calc_sbert_by_ref_indices(candid_emb)).cuda()
        acquisitions = self.GP_edit.acquisition(candid_emb_input, bias=self.bias, add_mean=add_mean)
        del candid_emb, candid_emb_input, add_mean
        return acquisitions

    def bayesian_step(self):
        '''
            return argmax_i AF(x_i)
        '''
        # generic bo step
        candidates = list(set(self.indices) - set(self.succ_indices) - set(self.stop_indices))
        acquisitions = self.calc_acquisitions_by_indices(candidates)
        max_ind = candidates[torch.argmax(acquisitions)]
        max_text = self.ui_pool[max_ind]

        # update trial counter
        self.update_trial_counter([max_ind])

        # + edit candidates
        prev_max_text = max_text
        for i in range(self.epsilon):
            max_text_candidates = self.synonym_func.get_sentence_candidates(max_text, self.num_synonym_candidates)
            max_text_candidates = self.filter_candidates(max_text_candidates)
            max_text_acquisitions = self.calc_acquisitions_by_emb(max_text_candidates)
            max_text = max_text_candidates[torch.argmax(max_text_acquisitions)]
            if max_text == prev_max_text: break
            prev_max_text = max_text
        return max_ind, max_text_candidates[torch.argmax(max_text_acquisitions)]
    
    def dpp_step(self, top_indices):
        num = min(len(top_indices), self.batch_size)
        Lmat = self.GP.get_covar(self.ui_input[top_indices])
        Lmat = Lmat / torch.mean(torch.abs(Lmat))
        assert Lmat.shape[0] >= num, f'{Lmat.shape}, {num}'
        Lmat = Lmat.detach().cpu().numpy()
        batch_indices_ids = dpp_sample(Lmat, k=num, T=0) if Lmat.shape[0] > num else list(range(num))
        batch_indices = list(top_indices[batch_indices_ids].cpu().numpy())
        return batch_indices

    def edit_step_batch(self, batch_indices, edit_step_policy='argmax'):
        '''
            edit step batch.
            edit_step_policy is one of 'argmax','sample'
        '''
        # + edit candidates
        batch_edit_texts = []
        batch_edit_indices = []
        for batch_ind in batch_indices:
            batch_text = self.ui_pool[batch_ind]
            prev_batch_text = batch_text
            for _ in range(self.epsilon):
                batch_text_candidates = self.synonym_func.get_sentence_candidates(batch_text, self.num_synonym_candidates)
                batch_text_candidates = self.filter_candidates(batch_text_candidates, filter_refs=batch_edit_texts)
                if len(batch_text_candidates) == 0:
                    continue
                batch_text_acquisitions = self.calc_acquisitions_by_emb(batch_text_candidates)
                if edit_step_policy == 'argmax':
                    batch_edit_id = int(torch.argmax(batch_text_acquisitions).item())
                elif edit_step_policy == 'sample':
                    batch_text_acquisitions[batch_text_acquisitions < 0] = 0
                    batch_edit_id = int(torch.multinomial(batch_text_acquisitions,1).item())
                batch_text = batch_text_candidates[batch_edit_id]
                if batch_text == prev_batch_text: break
                prev_batch_text = batch_text

            batch_edit_texts.append(batch_text)
            batch_edit_indices.append(batch_ind)
        return batch_edit_indices, batch_edit_texts

    def bayesian_step_batch(self):
        '''
            return batch indices corresponding to self.batch_type
        '''
        # BO step
        candidates = torch.LongTensor(list(set(self.indices) - set(self.succ_indices) - set(self.stop_indices)))
        if len(candidates) < self.batch_size: 
            return self.edit_step_batch(list(candidates.cpu().numpy()), 'argmax')
        acquisitions = self.calc_acquisitions_by_indices(candidates)

        if self.batch_type == 'dpp_posterior':
            # select batch indices considering DPP prior
            top_indices = candidates[torch.topk(acquisitions, min(200,len(acquisitions)))[1]]
            batch_indices = self.dpp_step(top_indices)
            edit_step_policy = 'argmax'
        elif self.batch_type == 'no':
            # return topk AF indices
            num = min(len(acquisitions), self.batch_size)
            inds = torch.topk(acquisitions, num)[1]
            batch_indices = list(candidates[inds].cpu().numpy())
            edit_step_policy = 'argmax'
        elif self.batch_type == 'sample':
            # sampling via prob propto AF(x_i)
            assert torch.all(acquisitions > -1e-8), f"index {torch.where(acquisitions <= -1e-8)} is negative"
            acquisitions[acquisitions < 0] = 0
            batch_indices = list(candidates[list(torch.multinomial(acquisitions, self.batch_size, replacement=False).cpu().numpy())].cpu().numpy())
            edit_step_policy = 'sample'
        elif self.batch_type == 'sample_v2':
            # sampling via prob propto AF(x_i)
            assert torch.all(acquisitions > -1e-8), f"index {torch.where(acquisitions <= -1e-8)} is negative"
            acquisitions[acquisitions < 0] = 0
            num = min(len(acquisitions), 1000)
            inds = torch.topk(acquisitions, num)[1]
            inds_inds = list(torch.multinomial(acquisitions[inds], self.batch_size, replacement=False).cpu().numpy())
            batch_indices = list(candidates[inds[inds_inds]].cpu().numpy())
            edit_step_policy = 'sample'
        
        # update trial counter
        self.update_trial_counter(batch_indices)
        # consider edit candidates of candidates
        del candidates
        return self.edit_step_batch(batch_indices, edit_step_policy)

    def update_trial_counter(self, indices):
        for ind in indices:
            self.trial_counter[ind] += 1
            if self.trial_counter[ind] >= self.trial_patience:
                self.stop_indices.append(ind)

    def clean_memory_cache(self,debug=False):
        gc.collect()
        torch.cuda.empty_cache()

    @property
    def eval_indices(self):
        return self.eval_DI.indices
    @property
    def eval_scores(self):
        return self.eval_DI.scores    
    @property
    def eval_texts(self):
        return self.eval_DI.texts
    @property
    def eval_embs(self):
        return self.eval_DI.embs
    @property
    def eval_embs_input(self):
        return self.eval_DI.embs_input

    @property
    def fit_indices(self):
        return self.fit_DI.indices
    @property
    def fit_scores(self):
        return self.fit_DI.scores    
    @property
    def fit_texts(self):
        return self.fit_DI.texts
    @property
    def fit_embs(self):
        return self.fit_DI.embs
    @property
    def fit_embs_input(self):
        return self.fit_DI.embs_input

    @property
    def succ_indices(self):
        return self.succ_DI.indices
    @property
    def succ_scores(self):
        return self.succ_DI.scores    
    @property
    def succ_texts(self):
        return self.succ_DI.texts
    @property
    def succ_embs(self):
        return self.succ_DI.embs
    @property
    def succ_embs_input(self):
        return self.succ_DI.embs_input
