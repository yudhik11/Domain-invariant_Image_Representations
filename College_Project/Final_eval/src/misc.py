import numpy as np
import scipy

def princomp(A):
    M = (A - np.mean(A.T, axis=1).reshape(1, -1)).T
    [latent, coeff] = np.linalg.eig(np.cov(M))
    score = np.dot(coeff.T, M) # projection of the data in the new space
    args = np.argsort(-latent)
    coeff = coeff[:, args]
    return coeff, score, latent

def normalizedKernel(K):
    '''Takes a kernel and returns its normalized
    verion to be used further'''
    if K.shape[0] != K.shape[1]:
        return
    DD = np.abs(np.diag(K))
    sqr_DD = np.sqrt(DD).reshape(1,-1)
    sq_DD2 = sqr_DD.T.dot(sqr_DD)
    K = K / sq_DD2
    return K

def formKernel(X1, X2, param):
    '''Constructs a kernel to be used for further tasks'''
    K = np.matmul(X1, X2.T)
    K = normalizedKernel(K)
    return K

def NormData(fts):
    sqr_fts = np.sqrt(np.sum(fts**2, 1))
    sqr_fts = np.matrix(sqr_fts).T
    sqr_fts = np.matlib.repmat(sqr_fts, 1, fts.shape[1])
    fts = fts / sqr_fts
    return scipy.stats.zscore(fts)