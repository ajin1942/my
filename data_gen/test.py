import numpy as np
import torch
from tqdm import tqdm

pose1=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
pose2=[1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8]
score1=[0.11,0.12,0.13,0.14]
score2=[0.21,0.22,0.23,0.24]

data_numpy=np.zeros((3,2,4,1))
data_numpy[0,0, :,0] = pose1[0::2]
data_numpy[1,0, :,0] = pose1[1::2]    #从1开始间隔取数，纵坐标
data_numpy[2, 0, :,0] = score1        #得分

data_numpy[0,1, :,0] = pose2[0::2]
data_numpy[1,1, :,0] = pose2[1::2]    #从1开始间隔取数，纵坐标
data_numpy[2,1, :,0] = score2       #得分
 # [[[[0.1 ]
 #   [0.3 ]
 #   [0.5 ]
 #   [0.7 ]]
 #
 #  [[1.1 ]
 #   [1.3 ]
 #   [1.5 ]
 #   [1.7 ]]]
 #
 #
 # [[[0.2 ]
 #   [0.4 ]
 #   [0.6 ]
 #   [0.8 ]]
 #
 #  [[1.2 ]
 #   [1.4 ]
 #   [1.6 ]
 #   [1.8 ]]]
 #
 #
 # [[[0.11]
 #   [0.12]
 #   [0.13]
 #   [0.14]]
 #
 #  [[0.21]
 #   [0.22]
 #   [0.23]
 #   [0.24]]]]
fp = np.zeros((2,3,2,4,1))
fp[:, :, 0:data_numpy.shape[1], :, :] = data_numpy
def cos_sim(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos = np.dot(a,b)/(a_norm * b_norm)
    return cos

paris = { 'kinetics': ((1, 0, 2), (3, 1, 2))}

# 'ntu/xview', 'ntu/xsub',  'kinetics'
datasets = {
    'kinetics'
}
N, C, T, V, M = fp.shape
for dataset in datasets:
    for v1, v2, v3 in tqdm(paris[dataset]):
        data1 = np.zeros((N, 3, T, V, M))
        data3 = np.zeros((N, 3, T, V, M))
        data1[:, :C, :, :, :] = data3[:, :C, :, :, :] = data_numpy
        data1[:, :, :, v1, :] = data1[:, :, :, v1, :] - data1[:, :, :, v2, :]  #BA
        data3[:, :, :, v3, :] = data3[:, :, :, v3, :] - data3[:, :, :, v2, :]  #BC
        for n in range(N):
            for t in range(T):
                for m in range(M):
        # fp[0, :,0, v2, 0] = cos_sim([data1[0, 0, 0, v1, 0], data1[0, 1, 0, v1, :0]],[data3[0, 0, 0, v3, 0], data1[0, 1, 0, v3, :0]])
                    c = np.array([data1[n, 0, t, v1, m],data1[n, 1, t, v1, m]])
                    a = np.array([data3[n, 0, t, v3, m],data1[n, 1, t, v3, m]])
                    fp[n, 0, t, v2, m] = cos_sim(c, a)
    print (fp)