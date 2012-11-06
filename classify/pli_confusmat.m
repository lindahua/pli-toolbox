function C = pli_confusmat(K, c1, c2, op)
%PLI_CONFUSMAT Confusion matrix
%
%   C = PLI_CONFUSMAT(K, c1, c2);
%
%       Computes the confusion matrix C. Here, C is a K-by-K matrix
%       defined by
%
%           C(i,j) = sum(c1 == i & c2 == j).
%
%       K is the number of classes.
%
%   C = PLI_CONFUSMAT(K, c1, c2, 'normalize');
%
%       Returns a confusion matrix that is normalized per row.
%       

%% argument checking

if ~(isscalar(K) && isnumeric(K) && K == fix(K) && K >= 2)
    error('pli_confusmat:invalidarg', ...
        'K is a positive integer with K >= 2.');
end

if ~(isnumeric(c1) && isreal(c1) && isvector(c1))
    error('pli_confusmat:invalidarg', ...
        'c1 should be a numeric vector.');
end

if ~(isnumeric(c2) && isreal(c2) && isvector(c2))
    error('pli_confusmat:invalidarg', ...
        'c2 should be a numeric vector.');
end

if ~isequal(size(c1), size(c2))
    error('pli_confusmat:invalidarg', ...
        'The sizes of c1 and c2 are inconsistent.');
end

if nargin >= 4
    if ~strcmpi(op, 'normalize')
        error('pli_confusmat:invalidarg', ...
            'The 4th argument can only be ''normalize''.');
    end
    to_nrm = 1;
else
    to_nrm = 0;
end


%% main

C = pli_intcount([K, K], c1, c2);
C = double(C);

if to_nrm
    C = bsxfun(@times, C, 1 ./ sum(C, 2));
end


