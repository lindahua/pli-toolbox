function r = pli_findbin(edges, x, is_sorted)
%PLI_FINDBIN Finds bin indices for given values
%
%   r = PLI_FINDBIN(edges, x);
%
%       Consider a series of bins delimited by the edges, then the
%       bin index for a value x is i, when edges(i) <= x < edges(i+1).
%       Here, edges should be a sorted vector.
%
%       The bin index is set 0, when x < edges(1), and 
%       it is numel(edges) + 1, when x >= edges(end).
%
%       The input x can be a matrix, and r will be a matrix of the same
%       size. The class r is int32.
%
%   r = PLI_FINDBIN(edges, x, is_sorted);
%
%       Specify whether x is sorted in ascending order. If this is the
%       case, the function will use a faster algorithm (O(n+K)) to find
%       bins for all values in x. Otherwise, a slower standard algorithm
%       of complexity O(n*K) wil be used. 
%
%       When omitted, is_sorted is set to be false.
%

%% argument checking

if ~(isnumeric(edges) && isreal(edges) && isvector(edges) && ...
        numel(edges) >= 2)
    error('pli_findbin:invalidarg', ...
        'edges should be a real vector with at least two elements.');    
end

if ~(isnumeric(x) && isreal(x) && ismatrix(x))
    error('pli_findbin:invalidarg', 'x should be a real matrix.');
end

if ~isa(x, class(edges))
    x = cast(x, class(edges));
end

if nargin < 3
    is_sorted = false;
else
    if ~isscalar(is_sorted)
        error('pli_findbin:invalidarg', 'is_sorted should be a scalar.');
    end
end


%% main

r = findbin_cimp(edges, x, logical(is_sorted));

