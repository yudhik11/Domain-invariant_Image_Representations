clear all;

param = Config(4,3);
% 4 => Caltech
% 3 => dslr
[Data, Labels] = DataLoader(param.DATA_DIR);

source_domain = param.source; 
target_domain = param.target; 

% Load data splits
splits = load(param.result_filename);
train_ids = splits.train;
test_ids = splits.test;

fprintf('Source Domain - %s, Target Domain - %s\n\n', ...
    param.domain_names{source_domain}, param.domain_names{target_domain});

% Store results:
n = param.num_trials;
telapsed = zeros(n,1);
accuracy = zeros(n,1);
pred_labels = cell(n,1);

source_domain_data = Data{source_domain};
target_domain_data = Data{target_domain};

source_domain_labels = Labels{source_domain};
target_domain_labels = Labels{target_domain};

for i = 1:n
    fprintf('       Iteration: %d / %d\n', i, n);
    data.train.source = source_domain_data(train_ids.source{i}, :);
    data.train.target = target_domain_data(train_ids.target{i}, :);
    data.test.target = target_domain_data(test_ids.target{i}, :);
    
    labels.train.source = source_domain_labels(train_ids.source{i});
    labels.train.target = target_domain_labels(train_ids.target{i});
    labels.test.target = target_domain_labels(test_ids.target{i});
    
    if param.dim < size(data.train.source, 2)
        P = pca([data.train.source; data.train.target; data.test.target], ...
            'Economy', false);
        data.train.source = data.train.source * P(:, 1:param.dim);
        data.train.target = data.train.target * P(:, 1:param.dim);
        data.test.target = data.test.target * P(:, 1:param.dim);
    end
    
    tstart = tic;
    [model_mmdt, W] = TrainMmdt(labels.train, data.train, param);
    telapsed(i) = toc(tstart);
    
    % model prediction
    [labels_output, score] = predict(model_mmdt, [data.test.target, ones(length(labels.test.target),1)]);
    accuracy(i) = sum(labels_output == labels.test.target') / length(labels.test.target);    

    fprintf('Accuracy = %.3f (Time = %6.2f)\n', accuracy(i), telapsed(i));
end
fprintf('\n\n Mean Accuracy = %.3f  (Mean time = %6.3f)\n', ...
    mean(accuracy)*100.0, mean(telapsed));