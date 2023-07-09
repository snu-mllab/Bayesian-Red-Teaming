import random
import numpy as np
import gc, torch
from tools.search.bayesopt.surrogate_model.gp_model_by_indices import MyGPModelByInd
from tools.search.bayesopt.dpp.dpp import dpp_sample, dpp_init_fixed
from tools.search.bayesopt.acquisition.algorithm.kmeanspp import kmeans_pp_cuda_cosine
from tools.slice_op import list_by_indices
from tools.bleu import BLEULauncher, SBERTLauncher, get_self_bleu
import copy
class BayesianQuestionChoicer:
    def __init__(
            self, 
            ui_pool, 
            ui_emb, 
            ui_tox, 
            expl_budget, 
            batch_type='dpp_posterior', 
            thre=0.0, 
            div_type='none',
            div_coeff=0.0,
            target_BLEU=None,
            use_sod=True, 
            use_tox_kernel=False,
            printer=print
        ):
        self.printer = printer
        self.counter = 0

        self.ui_pool = ui_pool
        self.ui_emb = ui_emb
        self.ui_tox = ui_tox
        self.expl_budget = expl_budget
        self.indices = list(range(len(self.ui_pool)))

        self.eval_indices = []
        self.eval_scores = []
        self.fit_indices = []
        self.fit_scores = []
        self.succ_indices = []

        self.batch_size = 10
        self.batch_type = batch_type
        self.thre = thre
        self.use_sod = use_sod
        self.use_tox_kernel = use_tox_kernel
        self.sod_size = 1000
        self.sod_limit = 10000
        self.memory_count = 0

        assert div_type in ['none','bleu','sbert','bleu_adapt'], f"div type should be one of [none, bleu, sbert] not {div_type}"
        self.div_type = div_type
        self.div_coeff = div_coeff
        self.target_BLEU = target_BLEU
        self.init_div()

        self.init_GP()

        self.bias = -1.0

    def clean(self):
        self.eval_indices = []
        self.eval_scores = []
        self.fit_indices = []
        self.fit_scores = []
        self.succ_indices = []
        self.batch_size = 10
        self.memory_count = 0
        self.init_div()
        self.init_GP()
    
    def init_proj_mat(self):
        self.hidden_dim = 200
        self.linear = torch.nn.Linear(self.ui_emb.shape[1], self.hidden_dim, bias=False).double().cuda()
        for p in self.linear.parameters():
            torch.nn.init.normal_(p, 0.0, 1./np.sqrt(self.hidden_dim))
        with torch.no_grad():
            self.ui_emb_input = self.linear(self.ui_emb.cuda())

    def init_GP(self):
        self.ui_emb_input = self.ui_emb.clone()

        if self.use_tox_kernel:
            assert self.ui_tox is not None
            self.ui_tox = torch.DoubleTensor(self.ui_tox)
            self.ui_emb_input = torch.cat([self.ui_emb_input, self.ui_tox.view(-1,1)], axis=1)

        self.GP = MyGPModelByInd(self.ui_emb_input, fit_iter=20) 

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

    def subset_of_data(self, indices):
        assert len(indices) > self.sod_size, "something wrong"
        _, selected_indices = kmeans_pp_cuda_cosine(
                                        data = self.ui_emb[indices], 
                                        k = self.sod_size
                                    )
        return selected_indices
    
    def subsample_fit_indices_by_sod(self):
        if len(self.eval_indices) > self.sod_limit:
            rnd_inds = random.sample(range(len(self.eval_indices)), self.sod_limit)
            indices = list_by_indices(self.eval_indices, rnd_inds)
            scores = list_by_indices(self.eval_scores, rnd_inds)
        else:
            indices = self.eval_indices
            scores = self.eval_scores
        selected_indices = self.subset_of_data(indices)
        self.fit_indices = list_by_indices(indices, selected_indices)
        self.fit_scores = list_by_indices(scores, selected_indices)

    def exploration(self):
        sampled_indices = random.sample(self.indices, self.expl_budget)
        return sampled_indices
    
    def update(self, indices, scores):
        # if we use sod and fitting indices size exceed sod limit, 
        # we run subset of data to reduce fitting indices size.
        self.clean_memory_cache()
        self.eval_indices.extend(indices)
        self.eval_scores.extend(scores)
        self.fit_indices.extend(indices)
        self.fit_scores.extend(scores)

        for ind, score in zip(indices, scores):
            if score > self.thre:
                self.succ_indices.append(ind)
                if self.div_type == 'sbert':
                    self.div_launcher.add_refs([self.ui_pool[ind]], self.ui_emb[ind].view(1,-1))

        if self.use_sod and len(self.eval_indices) > self.sod_size:
            self.subsample_fit_indices_by_sod()
        self.GP.fit(self.fit_indices, self.fit_scores)

        
        if self.counter % 10 == 0 and len(self.succ_indices) > 100:
            self.is_div_launcher_available = True
            if self.div_type == 'bleu' or self.div_type == 'bleu_adapt':
                cur_ref_indices = random.sample(self.succ_indices, 500) if len(self.succ_indices) >= 500 else self.succ_indices
                self.div_launcher.init_refs(list_by_indices(self.ui_pool, cur_ref_indices))
                self.div_vector = torch.DoubleTensor(self.div_launcher.calc_bleu_all()).cuda()
            elif self.div_type == 'sbert':
                assert len(self.succ_indices) == len(self.div_launcher.refs)
                indices = list(range(len(self.succ_indices)))
                cur_ref_indices = random.sample(indices, 500) if len(indices) >= 500 else indices
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
            self.bias += self.div_coeff * self.div_vector[self.eval_indices]
        self.bias = torch.max(self.bias)
        self.counter += 1

    def adapt_div_coeff(self):
        if self.is_div_launcher_available:
            # calc average self bleu
            cur_self_bleus = []
            for i in range(10):
                sampled_inds = random.sample(self.succ_indices, 100)
                cur_self_bleus.append(get_self_bleu(list_by_indices(self.ui_pool, sampled_inds)))
            cur_self_bleu = np.mean(cur_self_bleus)
            if cur_self_bleu > self.target_BLEU:
                self.div_coeff = min(self.adapt_max, self.div_coeff * self.adapt_step_size)
            elif cur_self_bleu < self.target_BLEU - self.adapt_gap:
                self.div_coeff = max(self.adapt_min, self.div_coeff / self.adapt_step_size)

    def calc_acquisitions(self, candidates):
        if self.div_type == 'none':
            add_mean = 0.0
        else:
            add_mean = self.div_coeff * self.div_vector[candidates]
        acquisitions = self.GP.acquisition(candidates, bias=self.bias, add_mean=add_mean)
        return acquisitions

    def bayesian_step(self):
        '''
            return argmax_i AF(x_i)
        '''
        candidates = list(set(self.indices) - set(self.eval_indices))
        acquisitions = self.calc_acquisitions(candidates)
        max_ind = candidates[torch.argmax(acquisitions)]
        return max_ind
    
    def dpp_step(self, top_indices):
        num = min(len(top_indices), self.batch_size)
        Lmat = self.GP.get_covar(top_indices)
        Lmat = Lmat / torch.mean(torch.abs(Lmat))
        assert Lmat.shape[0] >= num, f'{Lmat.shape}, {num}'
        Lmat = Lmat.detach().cpu().numpy()
        batch_indices_ids = dpp_sample(Lmat, k=num, T=0) if Lmat.shape[0] > num else list(range(num))
        batch_indices = list(top_indices[batch_indices_ids].cpu().numpy())
        return batch_indices

    def bayesian_step_batch(self):
        '''
            return batch indices corresponding to self.batch_type
        '''
        candidates = torch.LongTensor(list(set(self.indices) - set(self.eval_indices)))
        if len(candidates) < self.batch_size: return list(candidates.numpy())
        acquisitions = self.calc_acquisitions(candidates)

        if self.batch_type == 'dpp_posterior':
            # select batch indices considering DPP prior
            top_indices = candidates[torch.topk(acquisitions, min(200,len(acquisitions)))[1]]
            batch_indices = self.dpp_step(top_indices)
        elif self.batch_type == 'no':
            # return topk AF indices
            num = min(len(acquisitions), self.batch_size)
            inds = torch.topk(acquisitions, num)[1]
            batch_indices = list(candidates[inds].cpu().numpy())
        elif self.batch_type == 'sample':
            # sampling via prob propto AF(x_i)
            assert torch.all(acquisitions > -1e-8), f"index {torch.where(acquisitions <= -1e-8)} is negative"
            acquisitions[acquisitions < 0] = 0
            batch_indices = list(candidates[list(torch.multinomial(acquisitions, self.batch_size, replacement=False).cpu().numpy())].cpu().numpy())
        elif self.batch_type == 'sample_v2':
            # sampling via prob propto AF(x_i)
            assert torch.all(acquisitions > -1e-8), f"index {torch.where(acquisitions <= -1e-8)} is negative"
            acquisitions[acquisitions < 0] = 0
            num = min(len(acquisitions), 1000)
            inds = torch.topk(acquisitions, num)[1]
            inds_inds = list(torch.multinomial(acquisitions[inds], self.batch_size, replacement=False).cpu().numpy())
            batch_indices = list(candidates[inds[inds_inds]].cpu().numpy())

        elif self.batch_type == 'dpp_posterior_total':
            # select batch indices considering DPP prior over eval_indices
            top_indices = candidates[torch.topk(acquisitions, min(200,len(acquisitions)))[1]]
            num = min(len(top_indices), self.batch_size)
            if self.succ_indices:
                n_succ, n_top = len(self.succ_indices), len(top_indices)
                tot_indices = torch.LongTensor(self.succ_indices + list(top_indices))
                Lmat = self.GP.get_covar(tot_indices)
                Lmat = Lmat / torch.mean(torch.abs(Lmat))
                assert Lmat.shape[0] >= n_succ + num, f'{Lmat.shape}, {n_succ + num}'
                Lmat = Lmat.detach().cpu().numpy()
                batch_indices_ids = dpp_init_fixed(Lmat, k=num, fixed_indices=list(range(n_succ)))[0] if Lmat.shape[0] > n_succ + num else list(range(num))
                batch_indices = list(tot_indices[batch_indices_ids].cpu().numpy())
                assert len(batch_indices_ids) == num and all(not (x in self.eval_indices) for x in batch_indices)
            else:
                batch_indices = self.dpp_step(top_indices)
        return batch_indices

    def clean_memory_cache(self):
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    from tools.gen.gen_tool import get_embedding
    ui_pool = [f'q{i} {2*i} {3*i} ff bad' for i in range(20000)]
    ui_emb = get_embedding(ui_pool)
    #ui_emb = torch.randn(50000,768).double()
    expl_budget = 200
    BQC = BayesianQuestionChoicer(
                                    ui_pool=ui_pool,
                                    ui_emb=ui_emb,
                                    expl_budget=expl_budget,
                                    use_sod=True,
                                    div_type='bleu',
                                    div_coeff=0.5,
                                )
    expl_indices = BQC.exploration()
    BQC.update(expl_indices, np.random.rand(len(expl_indices)))

    # print("BATCH")
    # for batch_type in ['no','sample','dpp_posterior_total','dpp_posterior']:
    #     BQC.batch_type = batch_type
    #     print()
    #     print("batch_type", batch_type)
    #     for i in range(1):
    #         batch_indices = BQC.bayesian_step_batch()
    #         print(i, len(batch_indices))
    #         print(batch_indices)
    #         #BQC.update(batch_indices, np.random.randn(len(batch_indices)))

    for i in range(100):
        batch_indices = BQC.bayesian_step_batch()
        print(i, len(batch_indices))
        print(batch_indices)
        BQC.update(batch_indices, np.random.randn(len(batch_indices)))

    BQC.clean()
    expl_indices = BQC.exploration()
    print(len(expl_indices))
    print(expl_indices)

    BQC.update(expl_indices, np.random.randn(len(expl_indices)))
    print("INDIVIDUAL")
    for i in range(30):
        indiv_idx = BQC.bayesian_step()
        print(indiv_idx)
        BQC.update([indiv_idx], np.random.randn(1))
    
