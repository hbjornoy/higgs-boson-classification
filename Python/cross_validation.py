import numpy as np
def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def split_k(x,y,k_indices, k):
    #Given the data x, y, k_indices build by build_k_idiices, and k
    #Splits the data into training and test data, where the k-th fold is the test data
    x_test, y_test= x[k_indices[k]], y[k_indices[k]]
    train_ind=np.delete(k_indices,k,0)
    train_ind=np.ravel(train_ind)
    x_tr, y_tr= x[train_ind], y[train_ind]
    return x_test, y_test, x_tr, y_tr