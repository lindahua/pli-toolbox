function L = pli_dirichlet_logpdf(alpha, X, lnB)
%PLI_DIRICHLET_LOGPDF Log-pdf for Dirichlet distribution
%
%   L = PLI_DIRICHLET_LOGPDF(alpha, X);
%
%       Evaluates the log-pdf at the columns of X with respect to 
%       a Dirichlet distribution or multiple Dirichlet distributions 
%       with parameter alpha.
%
%   L = PLI_DIRICHLET_LOGPDF(alpha, X, lnB);
%
%       This statement also provides a pre-computed log-Beta value,
%       which should be equal to mvbetaln(alpha).
%

%% main

m = size(alpha, 2);

% get lnB

if nargin < 3
    lnB = pli_mvbetaln(alpha);
else
    if ~(isfloat(lnB) && isreal(lnB) && numel(lnB) == m)
        error('pli_dirichlet_logpdf:invalidarg', ...
            'lnB should be a real vector with length m.');
    end
end

% evaluate logpdf

L = (alpha - 1)' * log(X);

if m == 1
    L = L - lnB;
else
    L = bsxfun(@minus, L, lnB(:));
end


