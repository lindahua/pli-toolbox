function [sx, b, e, gs] = idxgroup(K, labels)
%IDXGROUP Group indices according to labels
%
%   [sx, b, e] = IDXGROUP(K, labels)
%
%       Group indices according to their labels. 
%
%       Specifically, given a vector of labels, it returns sx, b, and e
%       that specify the grouping, such that sx(b(k):e(k)) gives the 
%       indices whose corresponding labels are k, i.e.
%
%           inds_k = sx(b(k):e(k))
%           all( labels(inds_k) == k )  is true
%          
%       In addition, sx(b(k):e(k)) is sorted.
%
%       When there is no label that equals k, then b(k) = e(k) + 1.
%       Also, e(k) - b(k) + 1 equals sum(labels == k).
%       
%       Note: it is ok that some values in labels are out of [1, K], 
%       then those values are not counted, and numel(sx) may be smaller
%       than numel(labels).
%
%   [sx, b, e, gs] = IDXGROUP(K, labels)      
%
%       Additionally returns a cell array gs, where gs{k} is
%       sx(b(k):e(k)). When b(k) > e(k), gs{k} is empty.
%
%   Remarks
%   -------
%       This function is very useful for efficiently extracting
%       samples grouped by labels.
%
%       For example, let X be a sample matrix of size [d n], and
%       labels be a vector of length n, then to process the samples
%       in X group-by-group, one can write
%
%       [sx, b, e] = IDXGROUP(K, labels);
%       for k = 1 : K
%           Xk = X(:, b(k):e(k));  % Xk is the k-th group of samples
%           ... process Xk ....
%       end
%

%% argument checking

if ~(isnumeric(K) && isreal(K) && isscalar(K) && K >= 1)
    error('idxgroup:invalidarg', 'K should be a positive integer.');
end
K = int32(K);

if ~(isnumeric(labels) && isreal(labels) && isvector(labels) && ...
        ~issparse(labels))
    
    error('idxgroup:invalidarg', ...
        'labels should be a non-sparse numeric vector.');
end

%% main

[sx, b, e] = idxgroup_cimp(K, int32(labels) - 1);
b = b + 1;

if e(K) < numel(labels)
    sx = sx(1:e(K));
end

if nargout >= 4
    gs = cell(K, 1);
    for k = 1 : K
        gs{k} = sx(b(k):e(k));
    end    
end

