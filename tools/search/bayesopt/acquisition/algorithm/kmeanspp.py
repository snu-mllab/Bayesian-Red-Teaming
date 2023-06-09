import numpy as np
import torch
def euc_dist_sq(data1, data2):
    '''
    inputs:
        data1 - numpy array of data points (n1, d)
        data2 - numpy array of data points (n2, d)
    '''
    n1, d1 = data1.shape
    n2, d2 = data2.shape
    assert d1 == d2, f"the embedding dimension of data1, data2 are different {d1} != {d2}."
    d = d1
    c = np.reshape(data1,[n1,1,d]) - np.reshape(data2,[1,n2,d])
    dist_sq = np.sum(np.square(c),axis=2)
    return dist_sq

def euc_dist_sq_test():
    data1 = np.random.randn(2,4)
    data2 = np.random.randn(3,4)
    print("data1 : ", data1)
    print("data2 : ", data2)
    print("euc_dist_sq : ", euc_dist_sq(data1,data2))

def hamming_dist_sq(data1, data2):
    '''
    inputs:
        data1 - numpy array of data points (n1, d)
        data2 - numpy array of data points (n2, d)
    '''
    n1, d1 = data1.shape
    n2, d2 = data2.shape
    assert d1 == d2, f"the embedding dimension of data1, data2 are different {d1} != {d2}."
    d = d1
    c = (np.reshape(data1,[n1,1,d]) != np.reshape(data2,[1,n2,d])) * 1.0
    dist_sq = np.square(np.sum(c,axis=2))
    return dist_sq

def hamming_dist_sq_test():
    data1 = np.random.randint(5, size=[2,4])
    data2 = np.random.randint(5, size=[3,4])
    print("data1 : ", data1)
    print("data2 : ", data2)
    print("hamming_dist_sq : ", hamming_dist_sq(data1,data2))

def kmeans_pp(data, k, dist='euclidean', init_ind=None):
    '''
    initialized the centroids for K-means++
    inputs:
        data - numpy array of data points having shape (n, d)
        k - number of clusters (k <= n)
        dist - the name of metric
        init_ind - int (if None, random init index)
    '''
    ## initialize the centroids list and add
    ## a randomly selected data point to the list
    centroids = []
    selected_indices = []
    
    if init_ind is None:
        init_ind = np.random.randint(data.shape[0])
    centroids.append(data[init_ind, :])
    selected_indices.append(init_ind)
    
    if dist == 'euclidean':
        d_sq_func = euc_dist_sq
    elif dist == 'hamming':
        d_sq_func = hamming_dist_sq

    ## compute remaining centroids
    for _ in range(k - 1):
        all_indices = list(range(data.shape[0]))
        unselected_indices = list(set(all_indices) - set(selected_indices))

        d_sq_to_centroid = d_sq_func(data[unselected_indices], data[selected_indices])
        min_d_sq_to_centroid = np.min(d_sq_to_centroid, axis=1)
        if np.sum(min_d_sq_to_centroid)==0:
            break
        #prob = min_d_sq_to_centroid / np.sum(min_d_sq_to_centroid)
        #next_centroid_ind = all_indices[np.random.choice(unselected_indices, p=prob)]
        next_centroid_ind = unselected_indices[ 
                    np.argmax(min_d_sq_to_centroid)
                ]

        selected_indices.append(next_centroid_ind)

        centroids.append(data[next_centroid_ind, :])
    return np.array(centroids), selected_indices

def kmeans_pp_cuda_cosine(data, k, init_ind=None, batch_size=5000):
    '''
    initialized the centroids for K-means++
    inputs:
        data - torch Tensor of data points having shape (n, d)
        k - number of clusters (k <= n)
        dist - the name of metric
        init_ind - int (if None, random init index)
    '''
    ## initialize the centroids list and add
    ## a randomly selected data point to the list
    if k > data.shape[0]:
        raise RuntimeError
    centroids = []
    selected_indices = []
    data_cuda = data.cuda()
    
    if init_ind is None:
        init_ind = np.random.randint(data.shape[0])
    centroids.append(data[init_ind, :].view(1,-1))
    selected_indices.append(init_ind)
    
    cosine_calculator = torch.nn.CosineSimilarity(dim=2, eps=1e-6)


    ## compute remaining centroids
    d_to_prev_centroids = torch.empty(0).double().cuda()

    with torch.no_grad():
        for _ in range(k - 1):
            all_indices = list(range(data.shape[0]))
            unselected_indices = list(set(all_indices) - set(selected_indices))
            #data_cuda_unselected = data_cuda[unselected_indices]
            data_cuda_selected = data_cuda[selected_indices[-1]].view(1,-1).unsqueeze(0)

            d_to_cur_centroid_pt = []
            for sidx in range(0,len(data_cuda),batch_size):
                eidx = min(sidx + batch_size, len(data_cuda))
                dcuda_partial = data_cuda[sidx:eidx].unsqueeze(1)
                cur_cos_sim = cosine_calculator(dcuda_partial, data_cuda_selected)
                d_to_cur_centroid_pt.append(1 - cur_cos_sim)
                del cur_cos_sim, dcuda_partial
            del data_cuda_selected
            d_to_cur_centroid = torch.cat(d_to_cur_centroid_pt,axis=0)
            d_to_cur_centroids = torch.cat([d_to_prev_centroids, d_to_cur_centroid], axis=1)
            d_to_cur_centroids_unselected = d_to_cur_centroids[unselected_indices]

            min_d_to_centroid = torch.min(d_to_cur_centroids_unselected, axis=1)[0]
            if torch.sum(min_d_to_centroid)==0:
                break
            #prob = min_d_sq_to_centroid / np.sum(min_d_sq_to_centroid)
            #next_centroid_ind = all_indices[np.random.choice(unselected_indices, p=prob)]
            next_centroid_ind = unselected_indices[ 
                        torch.argmax(min_d_to_centroid)
                    ]

            selected_indices.append(next_centroid_ind)

            centroids.append(data[next_centroid_ind, :].view(1,-1))
            del d_to_prev_centroids, d_to_cur_centroids_unselected, d_to_cur_centroid
            d_to_prev_centroids = d_to_cur_centroids

    return torch.cat(centroids,dim=0).cpu(), selected_indices

def cosine_dist(data1, data2, batch_size=500):
    data1_cuda, data2_cuda = data1.cuda(), data2.cuda()
    cosine_calculator = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
    ds = []
    for sidx in range(0, len(data1), batch_size):
        eidx = min(sidx + batch_size, len(data1))
        ds2 = []
        for sidx2 in range(0, len(data2), batch_size):
            eidx2 = min(sidx2 + batch_size, len(data2))
            ds2.append(1 - cosine_calculator(data1_cuda[sidx:eidx].unsqueeze(1), data2_cuda[sidx2:eidx2].unsqueeze(0)))
        ds.append(torch.cat(ds2,dim=1))
    return torch.cat(ds, dim=0)

def kmeans_pp_test():
    import matplotlib.pyplot as plt

    ## 1. euc
    data = np.random.randn(100,2)
    centroids, selected_indices = kmeans_pp(data, 5, dist='euclidean')
    plt.scatter(data[:,0],data[:,1],label=0)
    plt.scatter(data[selected_indices,0],data[selected_indices,1],label=1)
    plt.savefig('kmeans_test1.png')
    plt.close()
    ## 2. hamming
    data = np.random.randint(20,size=[100,2])
    centroids, selected_indices = kmeans_pp(data, 5, dist='hamming')
    plt.scatter(data[:,0],data[:,1],label=0)
    plt.scatter(data[selected_indices,0],data[selected_indices,1],label=1)
    plt.savefig('kmeans_test2.png')

def kmeans_pp_test2():
    dl = []
    for i in range(10):
        dl.append(np.random.randn(100,512) + i*10)
    data = np.reshape(np.stack(dl), [1000,512])
    print(data.shape)
    losses = []
    import time
    tt = 0
    for i in range(10):
        print(i)
        t0 = time.time()
        centroids, selected_indices = kmeans_pp(data, 30, dist='euclidean')
        t1 = time.time()
        tt += t1 -t0
        loss = np.sum(np.min(euc_dist_sq(data, centroids), axis=1))
        losses.append(loss)
    print("time : ", tt/10)
    
    rnd_losses = []
    for i in range(10):
        indices = np.random.choice(data.shape[0], size=[30], replace=False)
        centroids = data[indices,:]
        loss = np.sum(np.min(euc_dist_sq(data, centroids), axis=1))
        rnd_losses.append(loss)

    print("kmeans", sum(losses)/ len(losses), losses)
    print("random", sum(rnd_losses)/ len(rnd_losses), rnd_losses)

def kmeans_pp_test3():
    import time
    K = 1000
    for N in [10]:
    # K = 100
    # for N in [10,100,1000]:
        # Generate 50 chunks
        dl = []
        for i in range(K):
            center = np.random.randn(1,512)
            dl.append(center*10 + 0.1*np.random.randn(N,512))
        data = torch.DoubleTensor(np.reshape(np.stack(dl), [K*N,512]))
        print(N, data.shape)

        # Run kmeans pp
        losses = []
        tt = 0
        for i in range(3):
            print(i)
            t0 = time.time()
            centroids, selected_indices = kmeans_pp_cuda_cosine(data, K)
            t1 = time.time()
            tt += t1 -t0
            cdist = cosine_dist(data, centroids)

            loss = float(torch.sum(torch.min(cdist, axis=1)[0]).cpu().numpy())
            losses.append(loss)
        print("time : ", tt/3)
        
        # Run random
        rnd_losses = []
        for i in range(3):
            indices = np.random.choice(data.shape[0], size=[K], replace=False)
            centroids = data[indices,:]
            loss = float(torch.sum(torch.min(cosine_dist(data, centroids), axis=1)[0]).cpu().numpy())
            rnd_losses.append(loss)

        print("kmeans", sum(losses)/ len(losses), losses)
        print("random", sum(rnd_losses)/ len(rnd_losses), rnd_losses)



if __name__ == '__main__':
    #euc_dist_sq_test()
    #hamming_dist_sq_test()
    #kmeans_pp_test()
    #kmeans_pp_test2()
    kmeans_pp_test3()