import numpy as np
from numpy.lib.format import open_memmap  #存储在磁盘上的二进制文件中的数组创建内存映射，内存映射文件用于访问磁盘上的大段文件，而无需将整个文件读入内存。


def cos_sim(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm * b_norm == 0:
        return  0.0
    else:
        return np.dot(a,b)/(a_norm * b_norm)

paris = {

    'kinetics': ((4, 0, 7), (8, 1, 11), (3, 2, 1), (4, 3, 2), (8, 4, 9), (6, 5, 1), (7, 6, 5), (11, 7, 12), (4, 8, 9),
                 (8, 9, 10), (4, 10, 1), (7, 11, 12), (11, 12, 13), (7, 13, 1), (16, 14, 1), (17, 15, 1), (0, 16, 2), (0, 17, 5))

}

sets = {
    'val'
}

# 'ntu/xview', 'ntu/xsub',  'kinetics'
datasets = {
    'kinetics'
}
# bone
from tqdm import tqdm

for dataset in datasets:
    for set in sets:
        print(dataset, set)
        data = np.load('E:/project/datasets/kinetics-skeleton/{}/{}_data_joint.npy'.format(dataset, set),mmap_mode='r') #mmap允许将大文件分成小段进行读写，而不是一次性将整个数组读入内存。
        N, C, T, V, M = data.shape
        fp_sp = open_memmap(
            'E:/project/datasets/kinetics-skeleton/{}/{}_data_angle.npy'.format(dataset, set),
            dtype='float32',
            mode='w+',
            shape=(N, 3, T, V, M))

        fp_sp[:, :C, :, :, :] = data
        for v1, v2, v3 in tqdm(paris[dataset]):
            if dataset != 'kinetics':
                v1 -= 1
                v2 -= 1
            data1 = np.zeros((N, 3, T, V, M))
            data3 = np.zeros((N, 3, T, V, M))
            data1[:, :C, :, :, :] = data3[:, :C, :, :, :] = data
            data1[:, :, :, v1, :] = data1[:, :, :, v1, :] - data1[:, :, :, v2, :]  # BA
            data3[:, :, :, v3, :] = data3[:, :, :, v3, :] - data3[:, :, :, v2, :]  # BC
            for n in range(N):
                for t in range(T):
                    for m in range(M):
                        c = np.array([data1[n, 0, t, v1, m], data1[n, 1, t, v1, m]])
                        a = np.array([data3[n, 0, t, v3, m], data1[n, 1, t, v3, m]])
                        fp_sp[n, 0, t, v2, m] = cos_sim(c, a)



