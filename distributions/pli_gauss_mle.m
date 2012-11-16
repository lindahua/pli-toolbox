function G = pli_gauss_mle(X, w, cf, op)
%PLI_GAUSS_MLE Maximum likelihood estimation of Gaussian distribution(s)
%
%   G = PLI_GAUSS_MLE(X);
%       
%       Maximum likelihood estimation of a Gaussian distribution.
%           
%   G = PLI_GAUSS_MLE(X, w);
%
%       Maximum likelihood estimation of a Gaussian distribution or
%       a collection of Gaussian distribution(s) with weighted data.
%
%   G = PLI_GAUSS_MLE(X, [], cform);
%   G = PLI_GAUSS_MLE(X, w, cform);
%
%       Maximum likelihood estimation using specified form of covariance.
%
%   G = PLI_GAUSS_MLE(X, [], cform, 'tie-cov');
%   G = PLI_GAUSS_MLE(X, w, cform, 'tie-cov');
%       
%       Ties the covariance of different Gaussian components. 
%
%
%   Arguments
%   ---------
%   - X :       The matrix of input data, of size [d, n]. Each column
%               is a sample.
%
%   - w :       The weights. If w is omitted or empty, then the data
%               is not weighted (i.e. all samples has the same weights).
%               
%               weights can also be a matrix of size [n m] to specify
%               m groups of weights. In this case, m distributions are
%               estimated, each corresponding to a column of w.
%
%   - cform:    The form of covariance, which can take either of the
%               following values:
%               - 's':  scalar form
%               - 'd':  diagonal form
%               - 'f':  full form
%               When cform is omitted, it takes the default value 'f'.
%

%% argument checking

if ~(isfloat(X) && isreal(X) && ismatrix(X))
    error('pli_gauss_mle:invalidarg', 'X should be a real matrix.');
end

d = size(X, 1);
n = size(X, 2);

if nargin < 2 || isempty(w)
    w = ones(n, 1);
    
    m = 1;
    inv_sw = 1 / n;
    
else
    if ~(isfloat(w) && isreal(w) && ismatrix(w))
        error('pli_gauss_mle:invalidarg', ...
            'w should be either empty or a real matrix.');
    end
    
    if size(w, 1) ~= n
        error('pli_gauss_mle:invalidarg', ...
            'The sizes of X and w are inconsistent.');
    end
    
    m = size(w, 2); 
    sw = sum(w, 1);
    inv_sw = 1 ./ sw;
end

if nargin < 3
    cf = 'f';
else
    if ~(ischar(cf) && isscalar(cf))
        error('pli_gauss_mle:invalidarg', 'cform should be a character.');
    end
end

if nargin < 4
    tied = 0;
else
    if ~strcmpi(op, 'tie-cov')
        error('pli_gauss_mle:invalidarg', 'The 4th argument is invalid.');
    end
    tied = 1;
end

tie_multi = m > 1 && tied;
if tie_multi
    sw = sw ./ sum(sw);
end

%% main

% estimate mean

if m == 1
    mu = (X * w) * inv_sw;
else
    mu = bsxfun(@times, X * w, inv_sw);
end

% estimate covariance

switch cf
    case 's'
        inv_d = 1 / d;
        mx2 = sum(X.^2, 1) * inv_d;
        Exx = (mx2 * w) .* inv_sw;            
        Euu = sum(mu.^2, 1) * inv_d;
        cvals = (Exx - Euu).';
        
        if tie_multi
            cvals = sw * cvals;
        end
        
    case 'd'
        Exx = bsxfun(@times, (X.^2) * w, inv_sw);
        cvals = Exx - mu .^ 2;
        
        if tie_multi
            cvals = cvals * sw';
        end
        
    case 'f'
        if tie_multi
            cvals = zeros(d, d);
            for k = 1 : m
                Ck = est_full_cov(X, mu(:,k), w(:,k), inv_sw(k));
                cvals = cvals + Ck * sw(k);
            end
        else
            if m == 1
                cvals = est_full_cov(X, mu, w, inv_sw);
            else
                cvals = zeros(d, d, m);
                for k = 1 : m
                    mu_k = mu(:, k);
                    cvals(:,:,k) = est_full_cov(X, mu_k, w(:,k), inv_sw(k));
                end
            end
        end
        
    otherwise
        error('pli_gauss_mle:invalidarg','Invalid value of cform.');
end

% make Gaussian struct

G = struct( ...
    'tag', 'gauss', ...
    'num', m, ...
    'dim', d, ...
    'cform', cf, ...
    'mu', mu, ...
    'cvals', cvals);


%% Auxiliary functions

function C = est_full_cov(X, mu, w, inv_sw)

Exx = (X * bsxfun(@times, w, X')) * inv_sw;
C = Exx - mu * mu';


