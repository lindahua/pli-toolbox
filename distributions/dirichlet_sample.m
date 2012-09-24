function X = dirichlet_sample(alpha, n)
%DIRICHLET_SAMPLE Sample from a Dirichlet distribution
%
%   X = dirichlet_sample(alpha, n);
%       
%       Draws n samples from a Dirichlet distribution with parameter
%       alpha.
%

%% argument checking

if ~(isfloat(alpha) && isreal(alpha) && size(alpha, 2) == 1)    
    error('dirichlet_sample:invalidarg', ...
        'alpha should be a real column vector.');
end

%% main

if n == 1
    X = randg(alpha);
    X = X * (1 / sum(X));
else
    X = randg(alpha(:, ones(1, n)));
    X = bsxfun(@times, X, 1 ./ sum(X, 1));
end


