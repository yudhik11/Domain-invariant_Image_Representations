import numpy as np
import scipy.io
import numpy.matlib
import scipy.stats
import sys
import time
import warnings

from train_mmdt import *
from dataloader import LoadOfficePlusCaltechData
from config import Config
from update_labels import UpdateLabelValues
from misc import princomp

sys.path.append('./liblinear-weights-2.21/python')
from liblinearutil import predict

warnings.filterwarnings("ignore")

def main(args):
    global param
    param = Config()
    elaps = 20
    param['num_trials'] = elaps
    n = param['num_trials']
    telapsed = np.zeros((n,1))
    accuracy = np.zeros((n,1))
    [Data, Labels] = LoadOfficePlusCaltechData( param['DATA_DIR'])
    source_domain = param['source']
    target_domain = param['target']

    # Load splits based on filename and update train and test ids
    splits = scipy.io.loadmat(param['result_filename'])
    train_ids = splits['train']
    test_ids = splits['test']
    train_ids_source = train_ids[0][0][0][0]
    train_ids_target = train_ids[0][0][1][0]
    test_ids_source = test_ids[0][0][0][0]
    test_ids_target = test_ids[0][0][1][0]

    # cur_models  : dictionary storing the models created based on grid search 
    # on the values of C_s and C_t for the source and target domains.
    src, tar = args[1], args[2]
    cur_models={}
    cur_models['src'] = src
    cur_models['tar'] = tar
    C_s = [0.05, 0.1, 0.01, 0.5]
    C_t = [1, 5, 0.5, 2.5]
    models = []
    best_model = None
    m_acc = -np.inf
    for c_s in C_s:
        for c_t in C_t:
            param['C_s'], param['C_t'] = c_s, c_t
            for i in range(0, elaps):
                # Data Loading
                data = {}
                data['train'] = {}
                data['test'] = {}
                print(i)
                data['train']['source'] = Data[source_domain][train_ids_source[i]-1][0]
                data['train']['target'] = Data[target_domain][train_ids_target[i]-1][0]
                data['test']['target'] = Data[target_domain][test_ids_target[i]-1][0]
                labels = {}
                labels['train'] = {}
                labels['test'] = {}
                labels['train']['source'] = Labels[source_domain][train_ids_source[i]-1][0]
                labels['train']['target'] = Labels[target_domain][train_ids_target[i]-1][0]
                labels['test']['target'] = Labels[target_domain][test_ids_target[i]-1][0]
                labels = UpdateLabelValues(labels, param)
                
                if param['dim'] < np.shape(data['train']['source'])[1]:
                    arr = np.array(data['train']['source'])
                    arr = np.vstack((arr, data['train']['target'])) 
                    arr = np.vstack((arr, data['test']['target']))
                    
                    # PCA to reduce diensionality of data
                    # We pick first 20 components
                    P, _, _ = princomp(arr)
                    data['train']['source'] = np.matmul(data['train']['source'], P[:, :20])
                    data['train']['target'] = np.matmul(data['train']['target'], P[:, :20])
                    data['test']['target'] = np.matmul(data['test']['target'], P[:, :20])
                
                # Main function call for getting the trained models
                [model_mmdt, W] = TrainMmdt(labels['train'], data['train'], param)
                arg1 = data['test']['target']
                arg2 = np.ones((np.size(labels['test']['target']),1))
                arg3 = np.concatenate((arg1, arg2), 1)

                labels['test']['target'] = labels['test']['target'].reshape(-1,)
                [pl, acc, pe] = predict(labels['test']['target'], arg3, model_mmdt);
                accuracy[i] = acc[0];
                print('Accuracy = {} (Time = {})\n'.format(accuracy[i], telapsed[i]));
            print('Mean Accuracy = {} +/- {}  (Mean time = {})'.format(np.mean(accuracy), np.std(accuracy)/np.sqrt(elaps), np.mean(telapsed)))
            mod = {}
            mod['C_s'], mod['C_t'], mod['accuracy'] = c_s, c_t, np.mean(accuracy)
            models.append(mod)
            # Get the best model based on mean accuracy over elaps
            if(mod['accuracy'] > m_acc):
                m_acc = mod['accuracy']
                best_model = mod
    print(best_model)
    cur_models['models'] = models
    cur_models['best_model'] = best_model
param = {}

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python run.py src tar")
    main(sys.argv)
