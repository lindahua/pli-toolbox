function s = kmpp_seed(X, K)
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

%% argument checking

if ~(ismatrix(X) && isreal(X))
    error('kmpp_seed:invalidarg', 'X should be a real matrix.');
end
n = size(X, 2);

if ~(isscalar(K) && isreal(K) && K >= 1 && K < n)
    error('kmpp_seed:invalidarg', ...
        'K should be a positive integer with 1 <= K < n.');
end

%% main

i = randi(n, 1);

if K == 1
    s = i;
else
    s = zeros(1, K);
    s(1) = i;
    
    ds = sum(bsxfun(@minus, X, X(:,i)).^2, 1);
    
    for k = 2 : K
        p = ds * (1/sum(ds));
        i = ddsample(p, 1);        
        s(k) = i;        
        
        ds_k = sum(bsxfun(@minus, X, X(:,i)).^2, 1);
        ds = min(ds, ds_k);
    end
end
    


