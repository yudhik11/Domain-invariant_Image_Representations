#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.io
import scipy
import numpy.matlib
import scipy.stats
from sklearn.decomposition import PCA
import sys
import scipy.sparse as sps
import time
from sklearn.svm import SVC
import warnings


# In[2]:


def Config(source=None, target=None):
    '''
    Input::
        source(string) : source domain index (Default : webcam)
        target(string) : target domain index (Default : dslr)
    Output::
        param : Output dictionary containing parameters'''
    amazon = 0
    webcam = 1
    dslr = 2
    caltech = 3
    
    param = {}
    param['domains'] = [amazon, webcam, dslr, caltech]
    param['domain_names'] = ['amazon', 'webcam', 'dslr', 'caltech']
    param['use_Gaussian_kernel'] = False
    
    param['categories'] = ['back_pack', 'bike', 'calculator',
                           'headphones', 'keyboard', 'laptop_computer', 'monitor', 'mouse',
                           'mug', 'projector']
    
    param['DATA_DIR'] = './MaxMarginDomainTransforms/ToRelease_GFK/data/'
    param['held_out_categories'] = False
    if source is None:
        param['source'] = webcam
    else:
        param['source'] = source
    
    if target is None:
        param['target'] = dslr
    else:
        param['target'] = target
    
    param['num_trials'] = 20
    param['dim'] = 20
    param['norm_type'] = 'l2_zscore'
    param['C_s'] = 0.05
    param['C_t'] = 1
    param['mmdt_iter'] = 2
    
    if param['source'] == amazon:
        param['num_train_source'] = 20
    else:
        param['num_train_source'] = 8
    
    param['num_train_target'] = 3
    
    param['result_filename'] = './MaxMarginDomainTransforms/DataSplitsOfficeCaltech/SameCategory_{0}-{1}_{2}RandomTrials_10Categories.mat'.format(param['domain_names'][param['source']],
                                                                     param['domain_names'][param['target']],
                                                                     param['num_trials'])
    param['telapsed'] = {}
    return param


# In[3]:


def NormData(fts, norm_type='l2_zscore'):
    '''
    Input::
        fts(numpy.ndarray) : Data to be normalized
        norm_type(string) : The norm_type which should be computed
    '''
    sqr_fts = np.sqrt(np.sum(fts**2, 1))
    sqr_fts = np.matrix(sqr_fts).T
    sqr_fts = np.matlib.repmat(sqr_fts, 1, fts.shape[1])
    fts = fts / sqr_fts
    
    return scipy.stats.zscore(fts)


# In[4]:


def LoadOfficePlusCaltechData(foldername, norm_type):
    '''
    Input::
        foldername(string) : Name of folder containing data
        norm_type(string) : The norm_type which should be computed
    
    Output::
        Data(list) : Loaded data
        Labels(list) : Labels for the data'''
    domain_names = ['amazon_SURF_L10.mat', 'webcam_SURF_L10.mat', 'dslr_SURF_L10.mat', 'Caltech10_SURF_L10.mat']

    Data = []
    Labels = []
    for idx, name in enumerate(domain_names):
        fullfilename = foldername + name
        obj = scipy.io.loadmat(fullfilename)
        
        fts = obj['fts']
        labels = obj['labels']
        fts = NormData(fts, norm_type)
        Data.append(fts)
        Labels.append(labels)
        
    return Data, Labels


# In[ ]:


amazon = 0
webcam = 1
dslr = 2
caltech = 3
param = Config(0,2)


# In[ ]:


[Data, Labels] = LoadOfficePlusCaltechData( param['DATA_DIR'], param['norm_type'])


# In[ ]:


source_domain = param['source']
target_domain = param['target']

splits = scipy.io.loadmat(param['result_filename'])
train_ids = splits['train']
test_ids = splits['test']
# Data[source_domain]
train_ids_source = train_ids[0][0][0][0]
train_ids_target = train_ids[0][0][1][0]
test_ids_source = test_ids[0][0][0][0]
test_ids_target = test_ids[0][0][1][0]


# In[ ]:


n = param['num_trials']
telapsed = np.zeros((n,1))
accuracy = np.zeros((n,1))
pred_labels = np.zeros((n, 1))


# In[ ]:


def AugmentWithOnes(data):
    '''
    Input::
        data(np.ndarray) : Input np array'''
    return np.hstack((data, np.ones(((data.shape[0], 1)))))


# In[ ]:


def TrainMmdt(labels, data, param):
    '''
    Input::
        labels(dictionary) : Contains labels for the data
        data(np.ndarray) : ndarray containing the data
        param(dictionary) : Dictionary containing parameters for processing
    Output::
        model : Returns trained model
        A : Transformation matrix'''
    if ('C_s' not in param or 'C_t' not in param):
        param['C_s'] = 1
        param['C_t'] = 1
    
    if 'gamma' not in param:
        param['gamma'] = 1e-4
    
    dA = data['source'].shape[1]
    dB = data['target'].shape[1]
    
    param['A'] = np.eye(dB + 1, dA + 1)

    
    if 'train_classes' not in param:
        param['train_classes'] = np.sort(np.unique(labels['source']))
    
    for idx in range(param['mmdt_iter']):
        [model, data, param] = TrainMmdtOneIter(labels, data, param)
    

    A = param['A']
    return [model, A]

def get_accuracy(param):
    '''
    Input::
        param(dictionary) : contains required parameters
    Output::
        acc(float) : Returns the accuracy of prediction
    '''
    plabels = param['plabels']
    labels = param['labels_out'].reshape(-1,)
    acc = np.sum(plabels==labels)/len(labels)
    return acc

warnings.filterwarnings("ignore")
    
def princomp(A):

    M = (A - np.mean(A.T, axis=1).reshape(1, -1)).T
    [latent, coeff] = np.linalg.eig(np.cov(M))
    score = np.dot(coeff.T, M) # projection of the data in the new space
    args = np.argsort(-latent)
    eig = coeff[:, args]
    return coeff, score, latent


# In[ ]:


def TrainMmdtOneIter(labels, data, param):
    '''
    Input:
    labels : '''
    data['transformed_target'] =  np.matmul(AugmentWithOnes(data['target']),param['A'])
    data_svm = np.vstack((AugmentWithOnes(data['source']), data['transformed_target']))

    labels_svm = np.hstack((labels['source'].T, labels['target'].T))
    
    weights_s = param['C_s']* np.ones((labels['source'].size, 1))
    weights_t = param['C_t']* np.ones((labels['target'].size, 1))
    param['weights'] = np.vstack((weights_s, weights_t))
    
 
    model = SVC()
    model.fit(np.abs(data_svm), (labels_svm.T))

    tstart = time.time()

    param['telapsed']['idx'] = time.time() - tstart;
    return [model, data, param]


# In[ ]:


elaps = 20
for i in range(0, elaps):
    data = {}
    data['train'] = {}
    data['test'] = {}
    data['train']['source'] = Data[source_domain][train_ids_source[i]-1][0]
    data['train']['target'] = Data[target_domain][train_ids_target[i]-1][0]
    data['test']['target'] = Data[target_domain][test_ids_target[i]-1][0]
    labels = {}
    labels['train'] = {}
    labels['test'] = {}
    
    labels['train']['source'] = Labels[source_domain][train_ids_source[i]-1][0]
    labels['train']['target'] = Labels[target_domain][train_ids_target[i]-1][0]
    labels['test']['target'] = Labels[target_domain][test_ids_target[i]-1][0]

    
    if param['dim'] < np.shape(data['train']['source'])[1]:
        arr = np.array(data['train']['source'])
        arr = np.vstack((arr, data['train']['target'])) 
        arr = np.vstack((arr, data['test']['target']))
        
        P, _, _ = princomp(arr)

        data['train']['source'] = np.matmul(data['train']['source'], P[:, :20])
        data['train']['target'] = np.matmul(data['train']['target'], P[:, :20])
        data['test']['target'] = np.matmul(data['test']['target'], P[:, :20])
        

    [model_mmdt, W] = TrainMmdt(labels['train'], data['train'], param)

    arg1 = data['test']['target']
    arg2 = np.ones((np.size(labels['test']['target']),1))

    arg3 = np.concatenate((arg1, arg2), 1)
    plabels= model_mmdt.predict(np.abs(arg3))
    param['plabels'], param['labels_out'] = plabels, labels['test']['target']
    accuracy[i] = get_accuracy(param)
    print('Accuracy = {} \n'.format(accuracy[i]))

print('Mean Accuracy = {} '.format(np.mean(accuracy)))

