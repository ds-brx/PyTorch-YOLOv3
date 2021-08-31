import torch
import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix


def apply_weight_sharing(model):
    """
    Applies weight sharing to the given model
    """

    for name, module in model.named_modules():
        if type(module).__name__ == "Conv2d":
            dev = module.weight.device
            c_weight = module.weight.data.cpu().numpy()
            print(name)
            for ci, i in enumerate(c_weight):
                for cj, j in enumerate(i):
                    try:
                        shape = j.shape
                        mat = csr_matrix(j) if shape[0] < shape[1] else csc_matrix(j)
                        min_ = min(mat.data)
                        max_ = max(mat.data)
                        space = np.linspace(min_, max_, num=5)
                        kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1, precompute_distances=True, algorithm="full")
                        kmeans.fit(mat.data.reshape(-1,1))
                        new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
                        mat.data = new_weight
                        module.weight.data[ci][cj] = torch.from_numpy(mat.toarray()).to(dev)
                    except ValueError:
                        continue
    
