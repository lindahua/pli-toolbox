function L = pli_gauss_logpdf(G, X, ldcov)
%PLI_GAUSS_LOGPDF Evaluates log-pdf of Gaussian distribution(s).
%
%   L = PLI_GAUSS_LOGPDF(G, X);
%
%       Evaluates the log-pdf values at columns of X, with respect to
%       the Gaussian distribution(s) in G.
%
%   L = PLI_GAUSS_LOGPDF(G, X, ldcov);
%
%       Evaluates the log-pdf values, with pre-computed log-determinants
%       of the covariance matrix (or matrices).
%
%       Note: The function gauss_logdet can be used to compute the 
%       log-determinant of covariance.
%
%
%   Arguments
%   ---------
%   - G :       A Gaussian distribution struct.
%
%   - X :       The matrix of input samples, size = [d, n].
%               Here, d is the space dimension, and n is the number of 
%               samples.
%
%   - ldcov :   The pre-computed log-determinant of covariance(s).    
%
%   Returns
%   -------
%   - L :   The resultant matrix, size = [m, n].
%           Here, m is the number of models encapsulated in G, and
%           n is the number of samples.
%

%% main

sqD = pli_gauss_mahdist(G, X, 'sq');

if nargin < 3
    ldcov = pli_gauss_logdet(G);
else    
    if ~(isscalar(ldcov) || isequal(size(ldcov), [G.num 1]))
        error('pli_gauss_logpdf:invalidarg', ...
            'The size of ldcov is incorrect.');
    end
end

c = ldcov + G.dim * 1.8378770664093453;  % log(2 pi) == 1.837....

if isscalar(c)
    L = (-0.5) * (sqD + c);
else
    L = (-0.5) * bsxfun(@plus, sqD, c);
end

