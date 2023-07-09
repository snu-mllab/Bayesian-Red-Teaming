from sentence_transformers import SentenceTransformer
from config.path import DATASETPATH
import torch
from tools.file_op import read_pkl

from tools.bleu import diversity_bleu, get_self_bleu_fastest
import random 
import numpy as np
## set up tokenizer and BERT
print("loading model...")
sim_model = SentenceTransformer('all-distilroberta-v1', device='cuda')

def rm_dup(l):
    new_list = []
    for i, v in enumerate(l):
        if i % 1000 == 0:
            print( i, len(l))
        if v not in new_list and len(v)>0:
            new_list.append(v)
    return new_list

def div_all_in_one(ql, k=100, n_rep=10, return_only_sb=False):
    if len(ql) < k :
        if return_only_sb:
            return -1
        else:
            return -1, -1, -1
    self_bleus = []
    if not return_only_sb:
        div_sberts = []
        div_bleus = []
    for i in range(n_rep):
        sub_ql = random.sample(ql, k)
        if not return_only_sb:
            div_bleus.append(diversity_bleu(sub_ql))
            div_sberts.append(diversity_sbert(sub_ql))
        self_bleus.append(get_self_bleu_fastest(sub_ql))
    sb = np.mean(self_bleus)
    if not return_only_sb:
        db = np.mean(div_bleus)
        ds = np.mean(div_sberts)
        return sb, db, ds
    return sb
    
def best_subset_self_bleu(ql, k):
    opt_l = random.sample(ql,1)
    while len(opt_l) < k:
        print(len(opt_l))
        best_q = None
        best_val = 200
        for q in ql:
            if q not in opt_l:
                val = get_self_bleu_fastest(opt_l + [q])
                if best_val > val:
                    best_val = val
                    best_q = q
        opt_l.append(best_q)
    return get_self_bleu_fastest(opt_l)

def get_embedding(ql):
    emb = torch.DoubleTensor(sim_model.encode(ql,show_progress_bar=False))
    return emb
        
def get_embedding_test():
    gen_ui_l = read_pkl(DATASETPATH + 'gpt3_zero_shot_data_10000.pkl')
    ql = []
    for d in gen_ui_l:
        ql.extend(d['data'])
    emb = get_embedding(ql)
    print(emb.shape)
    print(emb.type())
    for i in range(10):
        print(torch.sum(emb[i]*emb[i]))
    '''
        (10000, 768)       
    '''

def diversity_sbert(texts):
    text_embs = get_embedding(texts)
    num_texts = len(text_embs)
    cosine_calcualtor = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
    distances = 1 - cosine_calcualtor(text_embs.unsqueeze(1), text_embs.unsqueeze(0))
    assert torch.all(torch.abs(torch.diagonal(distances,0))<1e-8), print(torch.diagonal(distances,0)[:10])
    sbert_score = torch.sum(distances) / (num_texts * (num_texts-1))
    sbert_score = float(sbert_score.numpy())
    return sbert_score

def average_min_distance_texts(hyp_texts, ref_texts, batch_size1=1000, batch_size2=1000):
    hyp_embs = get_embedding(hyp_texts)
    ref_embs = get_embedding(ref_texts)
    return average_min_distance_embs(hyp_embs, ref_embs, batch_size1, batch_size2)

def average_min_distance_embs(hyp_embs, ref_embs, batch_size1=1000, batch_size2=1000):
    cosine_calcualtor = torch.nn.CosineSimilarity(dim=2, eps=0.0).cuda()
    sum_min_dist = 0
    for sidx2 in range(0,len(hyp_embs),batch_size2):
        eidx2 = min(sidx2+batch_size2,len(hyp_embs))
        mds = []
        cur_hyp = hyp_embs[sidx2:eidx2].cuda()
        for sidx in range(0,len(ref_embs),batch_size1):
            eidx = min(sidx+batch_size1,len(ref_embs))
            cur_ref = ref_embs[sidx:eidx].cuda()
            distances = 1 - cosine_calcualtor(cur_hyp.unsqueeze(1), cur_ref.unsqueeze(0))
            mds.append(torch.min(distances,dim=1)[0])
        min_dists = torch.min(torch.stack(mds),dim=0)[0]
        sum_min_dist += torch.sum(min_dists)
    average_min_dist = float((sum_min_dist/len(hyp_embs)).cpu().numpy())
    return average_min_dist


examples = [
    "he was expecting it",
    "it was so simple",
    "he was hungry",
    "he knew it was coming",
    "he liked it",
    "he was starving",
    "he was hungry",
    "it was full",
    "it was too crowded",
    "it was dangerous"
]

baseline = [
    "they didn't know what they were doing",
    "they didn't want to lose",
    "they didn't know any better",
    "they didn't know what to expect",
    "they didn't have a chance to win",
    "they didn't have to",
    "they didn't know what to do with it",
    "they knew they were going to win",
    "they didn't want to miss the game",
    "they didn't want to miss a game",
]


if __name__ == "__main__":
    print(diversity_bleu(baseline))
    print(diversity_bleu(examples))     
    print(diversity_sbert(baseline))
    print(diversity_sbert(examples))   
    print(get_self_bleu_fastest(baseline))
    print(get_self_bleu_fastest(examples))
    #print(average_min_distance_texts(baseline,examples,batch_size2=2))
    #print(average_min_distance_texts(baseline,examples,batch_size2=10000))
    # print(average_min_distance_texts(examples,baseline))