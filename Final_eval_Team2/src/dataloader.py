import scipy.io
from misc import NormData

def LoadOfficePlusCaltechData(foldername):
    domain_names = ['amazon_SURF_L10.mat', 'webcam_SURF_L10.mat', 'dslr_SURF_L10.mat', 'Caltech10_SURF_L10.mat']
    Data = []
    Labels = []
    for idx, name in enumerate(domain_names):
        fullfilename = foldername + name
        obj = scipy.io.loadmat(fullfilename)
        
        fts = obj['fts']
        labels = obj['labels']
        fts = NormData(fts)
        Data.append(fts)
        Labels.append(labels)
        
    return Data, Labels