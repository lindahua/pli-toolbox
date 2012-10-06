function G = pli_makegauss(d, mu, cov, op)
%PLI_MAKEGAUSS Construct a Gaussian struct
%
%   G = PLI_MAKEGAUSS(d, mu, cov);
%
%       Constructs a struct representing a Gaussian distribution or
%       a collection of multiple Gaussian distribution(s) over a
%       d-dimensional space.
%
%   G = PLI_MAKEGAUSS(d, mu, cov, 'tie_cov');
%
%       Constucts a Gaussian struct with tied covariance,
%       i.e. a covariance matrix is shared across all distributions.
%
%   Arguments
%   ---------
%   - d :   The space dimension.
%
%   - mu :  The mean vector(s). mu can be either 0 (zero scalar) or
%           a matrix of size [d, m]. Here, m is the number of 
%           distributions. 
%
%   - cov : The covariance matrix (or matrices). When cov is a diagonal
%           matrix, a compact representation can be used. 
%           
%           When there is a single distribution or the covariance is
%           tied, cov can be in either of the following forms:
%
%           - a scalar : 
%               representing eye(cov) * d
%
%           - a column vector of size [d, 1] : 
%               representing diag(cov)
%
%           - a full covariance matrix of size [d, d]:
%
%           When cov is not tied, it can be in the following forms:
%
%           - a row vector of size [m, 1]:
%               representing multiple matrices as eye(cov(i)) * d
%
%           - a matrix of size [d, m]:
%               representing multiple matrices as diag(cov(:,i))
%
%           - a cube of size [d, d, m]
%               representing multiple covariance matrices as cov(:,:,i).
%
%   Returns
%   -------
%   - G :   The constructed Gaussian struct, which contains the following
%           fields:
%           - num :     The number of encapsulated distributions
%           - dim :     The dimension
%           - cform :   An integer that indicates the covariance form
%                       - 0 : scalar form
%                       - 1 : diagonal form
%                       - 2 : full matrix form
%           - mu :      The mean vector(s)
%           - cov :     The covariance(s)
%

%% argument checking

% for argument: d

if ~(isscalar(d) && d >= 1)
    error('pli_makegauss:invalidarg', 'd should be a positive integer.');
end

% for argument: mu

if isequal(mu, 0)
    m = 1;
else
    if ~(ismatrix(mu) && size(mu, 1) == d)
        error('pli_makegauss:invalidarg', 'The size of mu is invalid.');
    end
    m = size(mu, 2);
end

% for argument: op

if nargin < 4
    tie_c = 0;
else
    if ~strcmp(op, 'tie_cov')
        error('pli_makegauss:invalidarg', ...
            'Failed to recognize the 4th argument.');
    end
    tie_c = 1;
end

% for argument: cov

if ~(isfloat(cov) && isreal(cov))
    error('pli_makegauss:invalidarg', ...
        'cov should be a real matrix.');
end

cform = -1;

if m == 1 || tie_c
    
    if isscalar(cov)
        cform = 0;
    elseif ismatrix(cov)
        cov_siz = size(cov);
        if isequal(cov_siz, [d 1])
            cform = 1;
        elseif isequal(cov_siz, [d d])
            cform = 2;
        end
    end
    
else
    
    cov_siz = size(cov);
    if isequal(cov_siz, [m, 1])
        cform = 0;
    elseif isequal(cov_siz, [d, m])
        cform = 1;
    elseif isequal(cov_siz, [d, d, m])
        cform = 2;
    end
    
end

if cform < 0
    error('pli_makegauss:invalidarg', 'The size of cov is invalid.');
end
        
%% make struct

G = struct( ...
    'tag', 'gauss', ...
    'num', m, ...
    'dim', double(d), ...
    'cform', cform, ...
    'mu', mu, ...
    'cov', cov);


