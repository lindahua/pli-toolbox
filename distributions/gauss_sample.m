function X = gauss_sample(G, n)
%GAUSS_SAMPLE Samples from a Gaussian distribution
%
%   X = GAUSS_SAMPLE(G, n);
%       Draws n samples from a Gaussian distribution.
%
%   Arguments
%   ---------
%   - G :   A Gaussian struct with G.num == 1
%
%   - n :   The number of samples to draw
%

%% argument checking

if ~(isstruct(G) && strcmp(G.tag, 'gauss') && G.num == 1)
    error('gauss_sample:invalidarg', ...
        'G should be a Gaussian struct with G.num == 1.');
end

%% main

Z = randn(G.dim, n);

cov = G.cov;

switch G.cform
    case 0
        X = sqrt(cov) * Z;
    case 1
        if n == 1
            X = sqrt(cov) * Z;
        else
            X = bsxfun(@times, sqrt(cov), Z);
        end
    case 2
        X = chol(cov, 'lower') * Z;            
end

mu = G.mu;

if ~isequal(mu, 0)
    if n == 1
        X = X + mu;
    else
        X = bsxfun(@plus, X, mu);
    end
end

