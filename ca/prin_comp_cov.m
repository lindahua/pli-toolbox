function [P, vars, res] = prin_comp_cov(C, k)
%PRIN_COMP_COV Principal component analysis based on covariance
%
%   [P, vars, res] = PRIN_COMP_COV(C, k);
%
%       Performs principal componentn analysis based on the covariance
%       matrix C, and returns the first k principal components in U.
%       (Here, k >= 1).
%   
%   [P, vars, res] = PRIN_COMP_COV(C, p);
%
%       Performs principal components analysis based on the covariance
%       matrix C, and returns the first k principal components in U.
%       Here, k is determined such that the first k components preserve
%       at least ratio p of the variance. (Here, p < 1).
%
%   Arguments
%   ---------
%   - X :       The data matrix of size [d n]. Each column is a sample.
%
%   - k :       The number of principal components.
%
%   - p :       The ratio of variance to be preserved.
%
%  
%   Returns
%   -------
%   - P :       The matrix of principal eigenvectors, size = [d k].
%
%   - vars :    The variances of the principal components, size = [1 k].
%
%   - res :     The residual variance.
%
%   Note: you don't have to use all output arguments when calling this 
%   function. The function only performs the computation enough to
%   return the desired outputs.
%

%% argument checking

if ~(ismatrix(C) && isreal(C) && isfloat(C) && size(C,1) == size(C,2))
    error('prin_comp_cov:invalidarg', ...
        'C should be a real square matrix.');
end

if ~(isscalar(k) && isreal(k) && k > 0)
    error('prin_comp_cov:invalidarg', ...
        'The second argument is invalid.');
end

if k >= 1    
    if k ~= fix(k)
        error('prin_comp_cov:invalidarg', ...
            'k should be a positive integer.');
    end
    if k >= size(C, 1)
        error('prin_comp_cov:invalidarg', ...
            'k should be less than size(C, 1).');
    end
    fix_k = 1;
else
    fix_k = 0;
    p = k;
end


%% main

[U, evs] = symeig(C);

if ~fix_k
    cs = cumsum(evs);
    k = find(cs >= p * cs(end), 1);
end

P = U(:, 1:k);

if nargout >= 2
    vars = evs(1:k);
end

if nargout >= 3
    res = sum(evs(k+1:end));
end



