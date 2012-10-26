function T = pli_whiten(C, d, lambda)
%PLI_WHITEN Whitening transform
%
%   T = PLI_WHITEN(C);
%   T = PLI_WHITEN(C, d);
%   T = PLI_WHITEN(C, d, lambda);
%
%       Computes a whitening transform T, such that the transformed
%       covariance T * C * T' is (approximately) an identity matrix.
%
%   Arguments
%   ---------
%   - C :       The covariance matrix.
%   
%   - d :       The dimension of the transformed space. 
%               When d is not specified, it is determined to be the
%               approximate rank of C. 
%               
%   - lambda :  Regularization coefficent. (default = 0)
%

%% argument checking

if ~(isfloat(C) && ismatrix(C) && isreal(C) && size(C,1) == size(C,2))
    error('pli_whiten:invalidarg', 'C should be a real square matrix.');
end

d0 = size(C, 1);

if nargin < 2
    d = [];
else
    if ~(isnumeric(d) && isscalar(d) && d == fix(d) && d >= 1)
        error('pli_whiten:invalidarg', 'd should be a positive integer.');
    end
    if d > d0
        error('pli_whiten:invalidarg', 'd should not exceed size(C,1).');
    end
end

if nargin < 3
    lambda = 0;
else
    if ~(isfloat(lambda) && isreal(lambda) && isscalar(lambda) && lambda >= 0)
        error('pli_whiten:invalidarg', ...
            'lambda should be a non-negative real scalar.');
    end
end

%% main

[U, evs] = pli_symeig(C);

if isempty(d)
    d = find(evs >= 1.0e-6 * evs(1), 1, 'last');
end

if d < d0
    U = U(:, 1:d);
    evs = evs(1:d);
end

if lambda > 0
    evs = evs + lambda * evs(1);
end

T = bsxfun(@times, 1.0 ./ sqrt(evs), U.');

