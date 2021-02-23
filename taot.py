import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import adjusted_rand_score
from scipy.stats.mstats import zscore
from scipy.spatial.distance import cdist
import time


n_time=23
dim=7
tt = zscore(np.linspace(1,n_time,n_time), ddof=1).reshape(-1,1)
pp = np.ones((n_time,1), dtype=float)/n_time
ll = 12.5
ww = 3500000.0

delta = 1.0
lamda1 = 50.0
lamda2 = 1.0

'''
Order-Preserving Optimal Transport
'''
def _opw(v1, v2, max_iter=20, tolerance=0.005):
    ts1 = v1.reshape((n_time,dim))
    ts2 = v2.reshape((n_time,dim))
    N = ts1.shape[0]
    M = ts2.shape[0]
    
    P = np.zeros([N,M])
    mid_para = np.sqrt((1/(N**2) + 1/(M**2)))
    
    for i in range(1,N+1):
        for j in range(1,M+1):
            d = np.abs(i/N - j/M)/mid_para
            P[i-1,j-1] = np.exp(-d**2/(2*(delta**2)))/(delta*np.sqrt(2*np.pi))
    
    S = np.zeros([N,M]);
    for i in range(1,N+1):
        for j in range(1,M+1):
            S[i-1,j-1] = lamda1/((i/N-j/M)**2+1)
            
    D = cdist(ts1, ts2, 'sqeuclidean')
    D = D/np.median(D)
    
    K = np.multiply(P, np.exp((S - D)/lamda2))
    
    a = np.ones([N,1])/N
    b = np.ones([M,1])/M
    compt=0
    u = np.ones([N,1])/N 
    
    while (compt < max_iter):
        v = np.divide(b, np.matmul(K.T, u))
        u = np.divide(a, np.matmul(K, v))
        compt += 1
        if((compt%20 == 1) or (compt == max_iter)):
            v = np.divide(b, np.matmul(K.T, u))
            u = np.divide(a, np.matmul(K, v))
            criterion = np.sum(np.fabs(np.multiply(v, np.matmul(K.T, u))-b))
            if(criterion < tolerance):
                break
            compt += 1

    U = np.multiply(K, D)
    dist = np.sum(np.multiply(u, np.matmul(U, v)))
    return dist

'''
Time Adaptive Optimal Transport
'''
def _sinkhorn_ot(v1, v2, max_iter=5000, tol=0.005):
    
    ts1 = v1.reshape((n_time,dim))
    ts2 = v2.reshape((n_time,dim))
    M = euclidean_distances(ts1,ts2,squared=True) + ww*euclidean_distances(tt,tt,squared=True)
    # M = M/np.max(M)
    M = M/np.median(M)
    K = np.exp(-ll*M);
    U = np.multiply(K,M)
    
    uu = pp
    cur_iter = 0
    while (cur_iter < max_iter):
        vv = np.divide(pp, np.matmul(K.T, uu))
        uu = np.divide(pp, np.matmul(K, vv))
        cur_iter += 1
        if((cur_iter%20 == 1) or (cur_iter == max_iter)):
            vv = np.divide(pp, np.matmul(K.T, uu))
            uu = np.divide(pp, np.matmul(K, vv))
            cur_iter += 1
            criterion = np.sum(np.fabs(np.multiply(vv, np.matmul(K.T, uu))-pp))
            if(criterion < tol):
                break
    dist = np.sum(np.multiply(uu, np.matmul(U, vv)))
    return dist

'''
Euclidean Distance
'''
def _euclidean_squared(v1, v2):
    dist = np.sum(np.square(v1 - v2))
    return dist

'''
Dynamic Time Warping
'''
def _dynamic_time_warping(v1, v2):
    ts1 = v1.reshape((23,7))
    ts2 = v2.reshape((23,7))
    l1 = ts1.shape[0]
    l2 = ts2.shape[0]
    shape = (l2, l1)
    accumulated_cost = np.zeros(shape)
    pairwise_cost = euclidean_distances(ts2, ts1, squared=True)
#    pairwise_cost = np.zeros(shape)
    for row in range(l2):
        for col in range(l1):
            if(abs(row-col)>5):
                pairwise_cost[row,col] = 9999999999999
#            else:
#                pairwise_cost[row,col] = _euclidean_squared(ts1[col], ts2[row])
    
    accumulated_cost[0,0] = pairwise_cost[0,0]
    for col in range(1, l1):
        accumulated_cost[0,col] = accumulated_cost[0, col-1] + pairwise_cost[0,col]
    for row in range(1, l2):
        accumulated_cost[row,0] = accumulated_cost[row-1, 0] + pairwise_cost[row,0]
    for row in range(1, l2):
        for col in range(1, l1):
            accumulated_cost[row,col] = min([accumulated_cost[row-1,col-1],accumulated_cost[row,col-1],accumulated_cost[row-1,col]]) + pairwise_cost[row,col]
            
    return accumulated_cost[l2-1,l1-1]
    