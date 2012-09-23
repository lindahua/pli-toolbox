function G = gauss_mle(X, w, cform)
%GAUSS_MLE Maximum likelihood estimation of Gaussian distribution(s)
%
%   G = GAUSS_MLE(X);
%       
%       Maximum likelihood estimation of a Gaussian distribution.
%           
%   G = GAUSS_MLE(X, w);
%
%       Maximum likelihood estimation of a Gaussian distribution or
%       a collection of Gaussian distribution(s) with weighted data.
%
%   G = GAUSS_MLE(X, [], cform);
%   G = GAUSS_MLE(X, w, cform);
%
%       Maximum likelihood estimation using specified form of covariance.
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
%               - 's-tied': tied covariance in scalar form
%               - 'd-tied': tied covariance in diagonal form
%               - 'f-tied': tied covariance in full form
%
%               When cform is omitted, it takes the default value 'f'.
%

%% argument checking

if ~(isfloat(X) && isreal(X) && ismatrix(X))
    error('gauss_mle:invalidarg', 'X should be a real matrix.');
end

d = size(X, 1);
n = size(X, 2);

if nargin < 2 || isempty(w)
    w = ones(n, 1);
    
    m = 1;
    inv_sw = 1 / n;
    
else
    if ~(isfloat(w) && isreal(w) && ismatrix(w))
        error('gauss_mle:invalidarg', ...
            'w should be either empty or a real matrix.');
    end
    
    if size(w, 1) ~= n
        error('gauss_mle:invalidarg', ...
            'The sizes of X and w are inconsistent.');
    end
    
    m = size(w, 2); 
    sw = sum(w, 1);
    inv_sw = 1 ./ sw;
end

if nargin < 3    
    cf = 2;
    tied = 0;
else
    [cf, tied] = parse_cform(cform);
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
    case 0
        inv_d = 1 / d;
        mx2 = sum(X.^2, 1) * inv_d;
        Exx = (mx2 * w) .* inv_sw;            
        Euu = sum(mu.^2, 1) * inv_d;
        cov = (Exx - Euu).';
        
        if tie_multi
            cov = sw * cov;
        end
        
    case 1
        Exx = bsxfun(@times, (X.^2) * w, inv_sw);
        cov = Exx - mu .^ 2;
        
        if tie_multi
            cov = cov * sw';
        end
        
    case 2
        if tie_multi
            cov = zeros(d, d);
            for k = 1 : m
                Exx = est_full_cov(X, mu(:,k), w(:,k), inv_sw(k));
                mu_k = mu(:, k);
                Ck = Exx - mu_k * mu_k';
                cov = cov + Ck * sw(k);
            end
        else
            if m == 1
                cov = est_full_cov(X, mu, w, inv_sw);
            else
                cov = zeros(d, d, m);
                for k = 1 : m
                    mu_k = mu(:, k);
                    cov(:,:,k) = est_full_cov(X, mu_k, w(:,k), inv_sw(k));
                end
            end
        end
end

% make Gaussian struct

G = struct( ...
    'tag', 'gauss', ...
    'num', m, ...
    'dim', d, ...
    'cform', cf, ...
    'mu', mu, ...
    'cov', cov);


%% Auxiliary functions

function C = est_full_cov(X, mu, w, inv_sw)

Exx = (X * bsxfun(@times, w, X')) * inv_sw;
C = Exx - mu * mu';


function [cf, tied] = parse_cform(cform)

len = length(cform);

if len == 1
    tied = 0;
elseif len == 6 && strcmp(cform(3:6), 'tied')
    tied = 1;
else
    error('gauss_mle:invalidarg', 'Invalid value for cform.');
end

switch cform(1)
    case 's'
        cf = 0;
    case 'd'
        cf = 1;
    case 'f'
        cf = 2;
    otherwise
        error('gauss_mle:invalidarg', 'Invalid value for cform.');
end



    

