function s = kmpp_seed(X, K, costfun)
%KMPP_SEED Choose K seeds using K-means++ method
%
%   s = KMPP_SEED(X, K);
%       
%       Chooses K seeds from the samples in X, and returns 
%       their indices in s.
%
%       Here, K should be less than the number of samples, i.e.
%       size(X, 2).
%
%   s = KMPP_SEED(X, K, costfun);
%
%       Uses a customized cost function. By default, squared
%       distance is used. 
%
%       costfun(x, Y) should return an 1 x n cost vector, when
%       x is a column, and Y have n columns.
%

%% argument checking

if ~(ismatrix(X) && isreal(X))
    error('kmpp_seed:invalidarg', 'X should be a real matrix.');
end
n = size(X, 2);

if ~(isscalar(K) && isreal(K) && K >= 1 && K < n)
    error('kmpp_seed:invalidarg', ...
        'K should be a positive integer with 1 <= K < n.');
end

if nargin < 3 || isempty(costfun)
    use_costfun = 0;
else
    if ~isa(costfun, 'function_handle')
        error('kmpp_seed:invalidarg', ...
            'costfun should be a function handle.');
    end
    use_costfun = 1;
end

%% main

i = randi(n, 1);

if K == 1
    s = i;
else
    s = zeros(1, K);
    s(1) = i;
    
    if use_costfun
        ds = costfun(X(:,i), X);
    else        
        ds = sum(bsxfun(@minus, X, X(:,i)).^2, 1);
    end
    
    for k = 2 : K
        p = ds * (1/sum(ds));
        i = ddsample(p, 1);        
        s(k) = i;        
        
        if use_costfun
            ds_k = costfun(X(:,i), X);
        else
            ds_k = sum(bsxfun(@minus, X, X(:,i)).^2, 1);
        end        
        ds = min(ds, ds_k);
    end
end
    


