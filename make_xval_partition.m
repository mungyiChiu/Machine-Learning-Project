function [part] = make_xval_partition(n, n_folds)
% MAKE_XVAL_PARTITION - Randomly generate cross validation partition.
%
% Usage:
%
%  PART = MAKE_XVAL_PARTITION(N, N_FOLDS)
%
% Randomly generates a partitioning for N datapoints into N_FOLDS equally
% sized folds (or as close to equal as possible). PART is a 1 X N vector,
% where PART(i) is a number in (1...N_FOLDS) indicating the fold assignment
% of the i'th data point.

% YOUR CODE GOES HERE

%1.1
%part = randi(n_folds,[1 n]);


part = zeros(1, n);
i = 1;

while min(part) == 0
    index = randi(n);
    
    while part(index) ~= 0
        index = randi(n);
    end
    
    part(index) = i;
    
    if i == n_folds
        i = 1;
    else
        i = i + 1;
    end
end





    