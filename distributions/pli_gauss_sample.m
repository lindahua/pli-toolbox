function X = pli_gauss_sample(G, n)
%PLI_GAUSS_SAMPLE Samples from a Gaussian distribution
%
%   X = PLI_GAUSS_SAMPLE(G, n);
%       Draws n samples from a Gaussian distribution.
%
%   Arguments
%   ---------
%   - G :   A Gaussian struct with G.num == 1
%
%   - n :   The number of samples to draw
%


%% main

Z = randn(G.dim, n);

cvals = G.cvals;

switch G.cform
    case 's'
        X = sqrt(cvals) * Z;
    case 'd'
        if n == 1
            X = sqrt(cvals) * Z;
        else
            X = bsxfun(@times, sqrt(cvals), Z);
        end
    case 'f'
        X = chol(cvals, 'lower') * Z;            
end

mu = G.mu;

if ~isequal(mu, 0)
    if n == 1
        X = X + mu;
    else
        X = bsxfun(@plus, X, mu);
    end
end

