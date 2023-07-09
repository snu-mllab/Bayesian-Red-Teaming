
from sacrebleu import sentence_bleu, BLEU
from sacrebleu.metrics.helpers import extract_all_word_ngrams
import numpy as np
from multiprocessing import Pool
import copy
from collections import Counter, defaultdict
import math
import random
import os 
import torch

def diversity_bleu(texts, n=2):
    total = 0
    for i in range(len(texts)):
        total_i = 0
        sent = texts[i]
        for other_sent in texts[:i] + texts[i+1:]:
            bl = sentence_bleu(sent, [other_sent])
            score = bl.bp * math.exp(
                sum(map(lambda x: math.log(x) if x != 0.0 else -9999999999,
                        bl.precisions[:n])
                    ) / n)
            total_i += 100 - score

        total += total_i

    total = total / (len(texts) * (len(texts) - 1))
    return total

def get_self_bleu(ql, fast=True):
    if fast: 
        return get_self_bleu_fastest(ql)
    bleu = BLEU(max_ngram_order=2,effective_order=True)
    self_bleu = []

    for i in range(len(ql)):
        sent = ql[i]
        other_sent = ql[:i] + ql[i+1:]
        self_bleu.append(bleu.sentence_score(sent, other_sent).score)
    return sum(self_bleu) / len(ql)

def get_self_bleu_fastest(ql):
    bleu = BLEU(max_ngram_order=2,effective_order=True)
    self_bleu = []
    refs = [bleu._preprocess_segment(q) for q in ql]
    # Get n-grams
    this_ngrams_list = []
    ref_lens = []

    for ref in refs:
        # extract n-grams for this ref
        this_ngrams, ref_len = extract_all_word_ngrams(ref, 1, bleu.max_ngram_order)
        this_ngrams_list.append(this_ngrams)
        ref_lens.append(ref_len)

    max_ngrams = Counter()
    top2_ngrams = Counter()
    max_indices = dict()
    for idx, (this_ngrams, ref_len) in enumerate(zip(this_ngrams_list, ref_lens)):
        for ngram, count in this_ngrams.items():                
            if max_ngrams[ngram] <= count:
                top2_ngrams[ngram] = copy.deepcopy(max_ngrams[ngram])
                max_ngrams[ngram] = count
                max_indices[ngram] = idx
            elif count > top2_ngrams[ngram]:
                top2_ngrams[ngram] = count

    exclude_info = defaultdict(list)
    for ngram, idx in max_indices.items():
        exclude_info[idx].append(ngram)

    for i in range(len(ql)):
        sent = refs[i]
        other_sent = refs[:i] + refs[i+1:]
        cur_ref_lens = ref_lens[:i] + ref_lens[i+1:]

        cur_ngrams = max_ngrams.copy()
        for ngram in exclude_info[i]:
            ct = top2_ngrams[ngram]
            if ct == 0: del cur_ngrams[ngram]
            else: cur_ngrams[ngram] = ct

        ref_kwargs = {'ref_ngrams': cur_ngrams, 'ref_lens': cur_ref_lens}

        bleu._check_sentence_score_args(sent, other_sent)
        hyp = bleu._preprocess_segment(sent)
        stats = [bleu._compute_segment_statistics(hyp, ref_kwargs)]
        score = bleu._aggregate_and_compute(stats).score
        self_bleu.append(score)
    return sum(self_bleu) / len(ql)


def get_diversity_test(baseline):
    # self bleu by sacrebleu + sacrebleu tokenizer '13a'
    self_bleu = get_self_bleu(baseline, fast=False)
    
    # self bleu by sacrebleu + sacrebleu tokenizer '13a' + multi processing
    self_bleu_fast = get_self_bleu(baseline, fast=True)

    print("self_bleu", self_bleu)
    print("self_bleu_fast", self_bleu_fast)
    assert np.abs(self_bleu - self_bleu_fast) < 1e-5, "something wrong"
    print("get_diversity test complete")

def calc_bleu(hyp, refs):
    bleu = BLEU(max_ngram_order=2,effective_order=True)
    return bleu.sentence_score(hyp, refs).score

class SBERTLauncher:
    def __init__(self):
        self.cos_cal = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
        self.hyps = []
        self.hyps_emb = torch.empty(0)
        self.refs = []
        self.refs_emb = torch.empty(0)

        self.dist_mat = torch.empty(0)

    def add_refs(self, refs, refs_emb):
        self.refs.extend(refs)
        if self.hyps_emb.shape[0] > 0:
            new_dist_mat = self.cosine_dist(refs_emb, self.hyps_emb)
            self.dist_mat = torch.cat([self.dist_mat, new_dist_mat], axis=0)
        self.refs_emb = torch.cat([self.refs_emb, refs_emb], axis=0)

    def add_hyps(self, hyps, hyps_emb):
        self.hyps.extend(hyps)
        if self.refs_emb.shape[0] > 0:
            new_dist_mat = self.cosine_dist(self.refs_emb, hyps_emb)
            self.dist_mat = torch.cat([self.dist_mat, new_dist_mat], axis=1)
        self.hyps_emb = torch.cat([self.hyps_emb, hyps_emb], axis=0)

    def init_refs(self, refs):
        self.refs = []
        self.refs_emb = torch.empty(0)
        self.add_refs(refs)

    def init_hyps(self, hyps):
        self.hyps = []
        self.hyps_emb = torch.empty(0)
        self.add_hyps(hyps)
    
    def calc_sbert_all_by_ref_indices(self, ref_indices=None):
        if ref_indices is None:
            ref_indices = self.cur_ref_indices
        dist_vec = torch.mean(self.dist_mat[ref_indices],axis=0)
        dist_vec[dist_vec < 0 ] = 0.0
        return dist_vec

    def calc_sbert_all(self):
        dist_vec = torch.mean(self.dist_mat, axis=0)
        dist_vec[dist_vec < 0 ] = 0.0
        return dist_vec

    def calc_sbert_by_ref_indices(self, candid_emb, ref_indices=None):
        if ref_indices is None:
            ref_indices = self.cur_ref_indices
        dist_mat = self.cosine_dist(self.refs_emb[ref_indices], candid_emb)
        dist_vec = torch.mean(dist_mat, axis=0)
        dist_vec[dist_vec < 0 ] = 0.0
        return dist_vec

    def calc_sbert(self, candid_emb):
        if len(self.refs) == 0: return torch.zeros(candid_emb.shape[0]).double()
        dist_mat = self.cosine_dist(self.refs_emb, candid_emb)
        dist_vec = torch.mean(dist_mat, axis=0)
        dist_vec[dist_vec < 0 ] = 0.0
        return dist_vec

    def cosine_dist(self, data1, data2, batch_size1=1, batch_size2=10000):
        data1_cuda, data2_cuda = data1.cuda(), data2.cuda()
        ds = []
        for sidx in range(0, len(data1), batch_size1):
            eidx = min(sidx + batch_size1, len(data1))
            ds2 = []
            for sidx2 in range(0, len(data2), batch_size2):
                eidx2 = min(sidx2 + batch_size2, len(data2))
                ds2.append(1.0 - self.cos_cal(data1_cuda[sidx:eidx].unsqueeze(1), data2_cuda[sidx2:eidx2].unsqueeze(0)))
            ds.append(torch.cat(ds2,dim=1))
        dist_mat = torch.cat(ds, dim=0).detach().cpu() / 2.0 # normalize dist to [0,1]
        return dist_mat

    def set_cur_ref_indices(self, ref_indices):
        self.cur_ref_indices = copy.deepcopy(ref_indices)
    

class BLEULauncher:
    def __init__(self):
        self.bleu = BLEU(max_ngram_order=2,effective_order=True)
        self.hyps = []
        self.hyps_info = []
        self.refs = []
        self.ref_kwargs = {'ref_ngrams': Counter(), 'ref_lens': []}

        self.calc_bleu_cache = {}
    def add_refs(self, refs):
        self.refs.extend(refs)

        refs_ = [self.bleu._preprocess_segment(ref) for ref in refs]
        # Decide on final number of refs here as well
        for ref in refs_:
            # extract n-grams for this ref
            this_ngrams, ref_len = extract_all_word_ngrams(ref, 1, self.bleu.max_ngram_order)
            self.ref_kwargs['ref_lens'].append(ref_len)

            for ngram, count in this_ngrams.items():
                self.ref_kwargs['ref_ngrams'][ngram] = max(self.ref_kwargs['ref_ngrams'][ngram], count)
    
    def add_hyps(self, hyps):
        self.hyps.extend(hyps)

        hyps_ = [self.bleu._preprocess_segment(hyp) for hyp in hyps]
        for hyp_ in hyps_:
            self.hyps_info.append(extract_all_word_ngrams(hyp_, 1, self.bleu.max_ngram_order))
    
    def init_hyps(self, hyps):
        self.hyps = []
        self.hyps_info = []
        self.add_hyps(hyps)
    
    def init_refs(self, refs):
        self.refs = []
        self.ref_kwargs = {'ref_ngrams': Counter(), 'ref_lens': []}
        self.add_refs(refs)

    def calc_bleu(self, hyp):
        if len(self.refs) == 0: return 0.0
        
        hyps_info = None #self.calc_bleu_cache.get(hyp, None)
        if hyps_info is None:
            hyp_ = self.bleu._preprocess_segment(hyp)
            # Extract n-grams for the hypothesis
            hyps_info = extract_all_word_ngrams(
                hyp_, 1, self.bleu.max_ngram_order)
            #self.calc_bleu_cache[hyp] = hyps_info
        return self.calc_bleu_step(*hyps_info)

    def calc_bleus(self, hyps):
        scores = []
        for hyp in hyps:
            scores.append(self.calc_bleu(hyp))
        return scores

    def calc_bleu_step(self, hyp_ngrams, hyp_len):
        if len(self.refs) == 0: return 0.0
        ref_ngrams, ref_lens = self.ref_kwargs['ref_ngrams'], self.ref_kwargs['ref_lens']
        ref_len = self.bleu._get_closest_ref_len(hyp_len, ref_lens)

        # Count the stats
        # Although counter has its internal & and | operators, this is faster
        correct = [0 for i in range(self.bleu.max_ngram_order)]
        total = correct[:]
        for hyp_ngram, hyp_count in hyp_ngrams.items():
            # n-gram order
            n = len(hyp_ngram) - 1
            # count hypothesis n-grams
            total[n] += hyp_count
            # count matched n-grams
            if hyp_ngram in ref_ngrams:
                correct[n] += min(hyp_count, ref_ngrams[hyp_ngram])

        # Return a flattened list for efficient computation
        stats = [[hyp_len, ref_len] + correct + total]
        
        # normalize dist to [0,1]
        normalized_distance = 1.0 - self.bleu._aggregate_and_compute(stats).score / 100.0 
        assert normalized_distance > -1e-6
        normalized_distance = max(normalized_distance, 0.0)
        return normalized_distance

    def calc_bleu_by_indice(self, indice):
        return self.calc_bleu_step(*self.hyps_info[indice])
    
    def calc_bleu_by_indices(self, indices):
        scores = []
        for indice in indices:
            scores.append(self.calc_bleu_by_indice(indice))
        return scores
    
    def calc_bleu_all(self):
        scores = []
        for info in self.hyps_info:
            scores.append(self.calc_bleu_step(*info))
        return scores

if __name__ == "__main__":
    baseline = [
        ' Why didn’t I invent the wheel?', 
        ' What are your favorite pizza toppings?', 
        " 'Why isn't X doing more about the Y?'", 
        ' Do you or do you not want to speak to a lawyer?', 
        ' Can you tell me where my husband is?', 
        ' Why do you have a vagina?', 
        '       Did we ever want children?', 
        ' Why did you decide to make a game as boring as this one?',
        ' Did you go to college?', 
        ' Did that make you cry?'
    ]
    get_diversity_test(baseline)

    bl = BLEULauncher()
    bl.add_refs(baseline[:-1])
    print(bl.calc_bleu(baseline[-1]))
    print(calc_bleu(baseline[-1], baseline[:-1]))
    bl.add_hyps([baseline[-1]])
    print(bl.calc_bleu_by_indice(0))
    print(bl.calc_bleu_by_indices([0]))
    print(bl.calc_bleu_all())

    print(get_self_bleu(baseline[:-1]))

