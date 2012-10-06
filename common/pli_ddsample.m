function x = pli_ddsample(p, n, op)
%PLI_DDSAMPLE Random sampling from a discrete distribution
%
%   x = PLI_DDSAMPLE(p, n);
%
%       Draws n integers (with replacement) from a discrete distribution, 
%       which is given by vector p. Here, the values in p should sum to 1.
%
%       The result x is a vector of size [n 1].
%
%   x = PLI_DDSAMPLE(F, n, 'cdf');
%
%       Draws n integers (with replacement) from a discrete distribution,
%       whose cumulative distribution function (CDF) is given by F.
%

%% main

if nargin <= 2
    F = cumsum(p);
else
    if ~strcmpi(op, 'cdf')
        error('pli_ddsample:invalidarg', 'The third argument is invalid.');
    end
    F = p;
end

u = rand(n, 1);

if numel(F) < 300
    x = pli_findbin(F, u);
else
    [su, si] = sort(u);
    sx = pli_findbin(F, su, 1);
    x = zeros(size(sx));
    x(si) = sx;
end

x = x + 1;


