function D = gauss_mahdist(G, X, op)
%GAUSS_MAHDIST Evaluates Mahalanobis distances to Gaussian center(s)
%
%   D = GAUSS_MAHDIST(G, X);
%
%       Evaluates the Mahalanobis distances from columns of X to 
%       the centers of the Gaussian distribution(s) in G, with 
%       respect to the corresponding covariance(s).
%
%   D = GAUSS_MAHDIST(G, X, 'sq');
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

if ~(isstruct(G) && strcmp(G.tag, 'gauss'))
    error('gauss_mahdist:invalidarg', ...
        'G must be a Gaussian struct.');
end

d = G.dim;

if ~(isfloat(X) && isreal(X) && ismatrix(X) && size(X,1) == d)
    error('gauss_mahdist:invalidarg', ...
        'X must be a real matrix with size(X,1) == d.');
end

if nargin < 3
    sq = 0;
else
    if ~strcmpi(op, 'sq')
        error('gauss_mahdist:invalidarg', ...
            'The third input argument is not recognized.');
    end
    sq = 1;
end

%% main

f = G.cform;
m = G.num;

mu = G.mu;
cov = G.cov;

if m == 1
    if isequal(mu, 0)
        Z = X;
    else
        Z = bsxfun(@minus, X, mu);
    end
    
    if f == 0
        D = sum(Z.^2, 1) * (1 ./ cov);
    elseif f == 1
        D = sum(bsxfun(@times, Z.^2, 1 ./ cov), 1);
    else
        D = sum(Z .* (cov \ Z), 1);
    end    
    
else            
   if f == 0
       sxx = sum(X.^2, 1);
       suu = sum(mu.^2, 1);
       D = bsxfun(@plus, suu', sxx) - (2 * mu)' * X;
       if isscalar(cov)
           if cov ~= 1
               D = D * (1.0 / cov);
           end
       else
           D = bsxfun(@times, D, 1 ./ cov);
       end
       
   else
       if f == 1
           q = 1 ./ cov;
           if size(q, 2) == 1
               H = bsxfun(@times, q, mu);               
           else
               H = q .* mu;
           end
           sxx = q' * (X.^2); 
       
       else
           if ismatrix(cov)
               H = cov \ mu;
               sxx = sum(X .* (cov \ X), 1);
           else                                  
               sxx = zeros(size(X, 2), m);
               H = zeros(d, m);
               for k = 1 : m
                   Ck = cov(:,:,k);
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


