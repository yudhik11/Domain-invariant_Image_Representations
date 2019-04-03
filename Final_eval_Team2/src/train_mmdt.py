from assymetric_transform import learnAsymmTransformWithSVM
import numpy as np
import sys, time
import scipy.sparse as sps
sys.path.append('./liblinear-weights-2.21/python')
from liblinearutil import train

def TrainMmdt(labels, data, param):
    '''
    Main function for training the mmdt model and transformation
    matrix.
    
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
        [model, data, param] = TrainMmdtOneIter(labels, data, param, idx)
    
    model.weights = model.weights * param['A'].T
    A = param['A']
    return [model, A]

def TrainMmdtOneIter(labels, data, param, idx):
    '''
    Runs one training iteration on the data which comprises of the model
    training and then learning the transformation matrix based on the 
    predicted model parameters
    
    Input::
        labels(dictionary) : Contains labels for the data
        data(np.ndarray) : ndarray containing the data
        param(dictionary) : Dictionary containing parameters for processing
        idx : iteration number
    Output::
        model : Returns trained model
        A : Transformation matrix'''    
    data['transformed_target'] =  AugmentWithOnes(data['target'])*param['A']
    data_svm = np.vstack((AugmentWithOnes(data['source']), data['transformed_target']))

    labels_svm = np.hstack((labels['source'].T, labels['target'].T))
    
    weights_s = param['C_s']* np.ones((labels['source'].size, 1))
    weights_t = param['C_t']* np.ones((labels['target'].size, 1))
    param['weights'] = np.vstack((weights_s, weights_t))
    
    # Train model based on the weights which depend on hyperparameters C_s and C_t
    # Iteration Step 1
    model = train(list(param['weights']), list(labels_svm.T), sps.csr_matrix(data_svm), '-c 1 -q')
    
    
    tstart = time.time()
    weights = []
    for i in range(0, 210):
        weights.append(model.w[i])
    
    # Learn transformation matrix based on the weights obtained by training in step 1
    # Iteration Step 2
    L = learnAsymmTransformWithSVM(np.array(weights).reshape(10, -1), \
    param['train_classes'], AugmentWithOnes(data['target']), labels['target'], param)
    param['A'] = L.T
    param['telapsed']['idx'] = time.time() - tstart;
    model.weights = np.array(weights).reshape(10, -1)
    return [model, data, param]

def AugmentWithOnes(data):
    return np.hstack((data, np.ones(((data.shape[0], 1)))))