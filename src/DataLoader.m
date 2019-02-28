function [Data, Labels] = DataLoader(foldername)
    fname = '%s_SURF_L10.mat';
    domain_names = {'amazon', 'webcam', 'dslr', 'Caltech10'};

    Data = cell(numel(domain_names));
    Labels = cell(numel(domain_names));
    for d = 1:numel(domain_names)
       fullfilename = fullfile(foldername, sprintf(fname, domain_names{d}));
       load(fullfilename);
       fts = NormData(fts);
       Data{d} = fts;
       Labels{d} = labels';
    end
end

function fts = NormData(fts)
    fts = fts ./ repmat(sqrt(sum(fts.^2,2)),1,size(fts,2));
    % mean = 0 and variance = 1
    fts = zscore(fts,1); 
end
