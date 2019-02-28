function [model, A] = TrainMmdt(labels, data, param)

% Input:
%   labels.source, labels.target: label vectors for training data points
%   data.source, data.target: training data of the form - num_pts x num_features
%   param.C_s, param.C_t, param.mmdt_iter, param.train_classes (for use in
%   new category experiment setting)

% Output:
%   model - struct of the form in liblinear
%   A - transformation matrix learned

    if ~isfield(param, 'C_s') || ~isfield(param, 'C_t')
        param.C_s = 1;
        param.C_t = 1;
    end

    dA = size(data.source,2);
    dB = size(data.target,2);
    param.A = eye(dB+1, dA+1);

    if ~isfield(param, 'train_classes')
        param.train_classes = sort(unique(labels.source));
    end
    
    for iter = 1:param.mmdt_iter
        [model, data, param] = TrainMmdtOneIter(labels, data, param);
    end
    A = param.A;
end

%%
function [model, data, param] = TrainMmdtOneIter(labels, data, param)

    data.transformed_target = [data.target, ones(size(data.target,1),1)]*param.A;

    data_svm = [[data.source, ones(size(data.source,1),1)]; data.transformed_target];
    labels_svm = [labels.source, labels.target];

    weights_s = param.C_s * ones(length(labels.source), 1);
    weights_t = param.C_t * ones(length(labels.target), 1);
    param.weights = [weights_s; weights_t];

    model = fitcensemble(data_svm, labels_svm', 'Method','bag', 'W', ...
        param.weights/ sum(param.weights));

end