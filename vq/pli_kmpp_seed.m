function s = pli_kmpp_seed(X, K, costfun)
%PLI_KMPP_SEED Choose K seeds using K-means++ method
%
%   s = PLI_KMPP_SEED(X, K);
%       
%       Chooses K seeds from the samples in X, and returns 
%       their indices in s.
%
%       Here, K should be less than the number of samples, i.e.
%       size(X, 2).
%
%   s = PLI_KMPP_SEED(X, K, costfun);
%
%       Uses a customized cost function. By default, squared
%       distance is used. 
%
%       costfun(x, Y) should return an 1 x n cost vector, when
%       x is a column, and Y have n columns.
%

%% argument checking

if ~(ismatrix(X) && isreal(X) && ~issparse(X) && ~isempty(X))
    error('pli_kmpp_seed:invalidarg', ...
        'X should be a non-sparse real matrix.');
end
n = size(X, 2);

if ~(isscalar(K) && isreal(K) && K >= 1 && K <= n)
    error('pli_kmpp_seed:invalidarg', ...
        'K should be a positive integer with 1 <= K <= n.');
end

if nargin < 3 || isempty(costfun)
    use_costfun = 0;
else
    if ~isa(costfun, 'function_handle')
        error('pli_kmpp_seed:invalidarg', ...
            'costfun should be a function handle.');
    end
    use_costfun = 1;
end

%% main



if K == 1
    s = randi(n, 1);
    
else
    if ~use_costfun
        
        if ~isa(X, 'double')
            X = double(X);
        end
        
        s = kmpp_cimp(X, rand(1, K)); 
    else
        i = randi(n, 1);
        
        s = zeros(1, K);
        s(1) = i;
        
        ds = costfun(X(:,i), X);
        
        for k = 2 : K
            p = ds * (1/sum(ds));
            i = pli_ddsample(p, 1);
            s(k) = i;
            
            ds_k = costfun(X(:,i), X);
            ds = min(ds, ds_k);
        end        
        
    end

end
    


