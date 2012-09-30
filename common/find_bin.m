function r = find_bin(edges, x, is_sorted)
%FIND_BIN Finds bin indices for given values
%
%   r = FIND_BIN(edges, x);
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
%   r = FIND_BIN(edges, x, is_sorted);
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
    error('find_bin:invalidarg', ...
        'edges should be a real vector with at least two elements.');    
end

if ~(isnumeric(x) && isreal(x) && ismatrix(x))
    error('find_bin:invalidarg', 'x should be a real matrix.');
end

if ~isa(x, class(edges))
    x = cast(x, class(edges));
end

if nargin < 3
    is_sorted = false;
else
    if ~(islogical(is_sorted) && isscalar(is_sorted))
        error('find_bin:invalidarg', ...
            'is_sorted should be a logical scalar.');
    end
end


%% main

r = find_bin_cimp(edges, x, is_sorted);

