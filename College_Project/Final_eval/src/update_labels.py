def UpdateLabels(labels, all_categories, categories):
    for i in range(0, labels.size):
        oldL = labels[i]
        l = all_categories[oldL]
        newL = l.index(categories)
        labels[i] = newL;
    return labels

def UpdateLabelValues(labels, param):
    if 'all_categories' not in param:
        return labels
    all_categories = param['all_categories']
    categories = param['categories']
    labels['train']['source'] = UpdateLabels(labels['train']['source'], all_categories, categories)
    labels['train']['target'] = UpdateLabels(labels['train']['target'], all_categories, categories)
    labels['test']['target'] = UpdateLabels(labels['test']['target'], all_categories, categories)
    return labels