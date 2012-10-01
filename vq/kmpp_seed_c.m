function s = kmpp_seed_c(C, K)
%KMPP_SEED_C Choose K seeds using K-means++ with pre-computed costs
%
%   s = KMPP_SEED_C(X, K);
%       
%       Chooses K seeds from n samples, and returns 
%       their indices in s. Here, C is an n-by-n cost table.
%
%       Here, K should be less than the number of samples.
%

%% argument checking

if ~(ismatrix(C) && isreal(C) && size(C,1) == size(C, 2))
    error('kmpp_seed_c:invalidarg', 'C should be a real square matrix.');
end
n = size(C, 1);

if ~(isscalar(K) && isreal(K) && K >= 1 && K < n)
    error('kmpp_seed_c:invalidarg', ...
        'K should be a positive integer with 1 <= K < n.');
end

%% main

i = randi(n, 1);

if K == 1
    s = i;
else
    s = zeros(1, K);
    s(1) = i;
    
    ds = C(i, :);
    
    for k = 2 : K
        p = ds * (1/sum(ds));
        i = ddsample(p, 1);        
        s(k) = i;
        ds = min(ds, C(i, :));
    end
end
    


