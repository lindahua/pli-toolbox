function D = pli_gauss_mahdist(G, X, op)
%PLI_GAUSS_MAHDIST Evaluates Mahalanobis distances to Gaussian center(s)
%
%   D = PLI_GAUSS_MAHDIST(G, X);
%
%       Evaluates the Mahalanobis distances from columns of X to 
%       the centers of the Gaussian distribution(s) in G, with 
%       respect to the corresponding covariance(s).
%
%   D = PLI_GAUSS_MAHDIST(G, X, 'sq');
%
%       Evaluates the squared Mahalanobis distances.
%
%   Arguments
%   ---------
%   - G :   A Gaussian distribution struct.
%
%   - X :   The matrix of input samples, size = [d, n].
%           Here, d is the space dimension, and n is the number of 
%           samples.
%
%   Returns
%   -------
%   - D :   The resultant matrix, size = [m, n].
%           Here, m is the number of models encapsulated in G, and
%           n is the number of samples.
%

%% argument checking

d = G.dim;

if ~(isfloat(X) && isreal(X) && ismatrix(X) && size(X,1) == d)
    error('pli_gauss_mahdist:invalidarg', ...
        'X must be a real matrix with size(X,1) == d.');
end

if nargin < 3
    sq = 0;
else
    if ~strcmpi(op, 'sq')
        error('pli_gauss_mahdist:invalidarg', ...
            'The third input argument is not recognized.');
    end
    sq = 1;
end

%% main

cf = G.cform;
m = G.num;

mu = G.mu;
cvals = G.cvals;

if m == 1
    if isequal(mu, 0)
        Z = X;
    else
        Z = bsxfun(@minus, X, mu);
    end
    
    if cf == 's'
        D = sum(Z.^2, 1) * (1 ./ cvals);
    elseif cf == 'd'
        D = sum(bsxfun(@times, Z.^2, 1 ./ cvals), 1);
    else
        D = sum(Z .* (cvals \ Z), 1);
    end    
    
else            
   if cf == 's'
       sxx = sum(X.^2, 1);
       suu = sum(mu.^2, 1);
       D = bsxfun(@plus, suu', sxx) - (2 * mu)' * X;
       if isscalar(cvals)
           if cvals ~= 1
               D = D * (1.0 / cvals);
           end
       else
           D = bsxfun(@times, D, 1 ./ cvals);
       end
       
   else
       if cf == 'd'
           q = 1 ./ cvals;
           if size(q, 2) == 1
               H = bsxfun(@times, q, mu);               
           else
               H = q .* mu;
           end
           sxx = q' * (X.^2); 
       
       else
           if ismatrix(cvals)
               H = cvals \ mu;
               sxx = sum(X .* (cvals \ X), 1);
           else                                  
               sxx = zeros(size(X, 2), m);
               H = zeros(d, m);
               for k = 1 : m
                   Ck = cvals(:,:,k);
                   H(:,k) = Ck \ mu(:,k);
                   sxx(:, k) = sum(X .* (Ck \ X), 1);
               end
               sxx = sxx.';
           end
       end
       
       suu = sum(mu .* H, 1);
       D = bsxfun(@plus, suu', sxx) - (2 * H)' * X;
   end
end
D(D < 0) = 0;

if ~sq
    D = sqrt(D);
end


