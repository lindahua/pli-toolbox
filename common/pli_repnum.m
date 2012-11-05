function y = pli_repnum(x, r)
%PLI_REPNUM Repeating numbers for certain times
%
%   y = PLI_REPNUM(n, r);
%
%       Here, n is a positive integer, and r is a vector of non-negative
%       integers of length n.
%
%       This statement yields a vector y of size 1-by-sum(r), which
%       repeats i by r(i) times.
%
%   y = PLI_REPNUM(x, r);
%
%       Here, x is a vector of length n. In the resultant vector y, 
%       x(i) is repeated by r(i) times.
%

%% argument checking

if isnumeric(x) && isreal(x) && ~issparse(x)
    if isscalar(x)
        n = x;
        x = [];
    elseif isvector(x)
        n = length(x);
    end
end

if ~(isvector(r) && length(r) == n && ~issparse(r))
    error('pli_repnum:invalidarg', ...
        'r should be a non-sparse vector of length n.');
end

%% main

y = repnum_cimp(n, uint32(r));

if ~isempty(x)
    y = x(y);
end

