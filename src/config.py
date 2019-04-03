def Config(source=None, target=None):
    '''Initiazation of necessary parametrs for compuatation
    
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
    
    param['categories'] = ['back_pack', 'bike', 'calculator',
                           'headphones', 'keyboard', 'laptop_computer', 'monitor', 'mouse',
                           'mug', 'projector']
    
    param['DATA_DIR'] = '../data/original_data/'
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
    param['C_s'] = 0.05
    param['C_t'] = 1
    param['mmdt_iter'] = 2
    
    if param['source'] == amazon:
        param['num_train_source'] = 20
    else:
        param['num_train_source'] = 8
    
    param['num_train_target'] = 3
    
    param['result_filename'] = '../data/DataSplitsOfficeCaltech/SameCategory_{0}-{1}_{2}RandomTrials_10Categories.mat'.format(param['domain_names'][param['source']],
                                                                     param['domain_names'][param['target']],
                                                                     param['num_trials'])
    param['telapsed'] = {}
    return param