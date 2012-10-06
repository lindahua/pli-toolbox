function v = pli_mvbetaln(X)
%PLI_MVBETALN Logarithm of multivariate beta function.
%
%   v = PLI_MVBETALN(X);
%
%       Evaluates the logarithm of multivariate Beta function for 
%       each vector of X.
%
%       If X is a vector, it returns a scalar, otherwise if X is a 
%       matrix of size [m, n], it returns a row vector of length n,
%       where v(i) corresponds to X(:,i).
%

%% main

v = sum(gammaln(X)) - gammaln(sum(X));


