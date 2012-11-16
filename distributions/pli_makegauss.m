function G = pli_makegauss(d, mu, cform, cvals)
%PLI_MAKEGAUSS Construct a Gaussian struct
%
%   G = PLI_MAKEGAUSS(d, mu, cform, vals);
%       
%       Constructs a struct representing a Gaussian distribution or 
%       a set of Gaussian distribution(s) over a d-dimensional space.
%
%       Mean vectors
%       -------------
%       The mean vector mu can be input as a full vector or just a zero
%       scalar to indicate zero mean.
%
%       When multiple distributions are to be packed, mu must be a
%       d-by-m matrix, each column corresponding to one distribution. 
%
%
%       Covariance
%       -----------
%       Here, cform, an indicator of the form that the covariance values 
%       are stored, can take either of the following values:
%
%       - 's':  scalar form, use a scalar s to represent a covariance 
%               matrix in the form of s * eye(d).
%
%               cvals is either a scalar or a column vector of length m.
%
%       - 'd':  diagonal form, use a vector v to represent a diagonal
%               matrix in the form of diag(v).
%
%               cvals should be either a column vector, or a matrix of 
%               size d-by-m.
%
%       - 'f':  full form, use a d x d covariance matrix.
%       
%               cvals should be either a d-by-d matrix, or an array of
%               size d-by-d-by-m.
%
%       Note that it allows packing multiple distribution with different
%       mean vectors and a shared covariance.
%
%   Returns
%   -------
%   - G :   The constructed Gaussian struct, which contains the following
%           fields:
%           - tag :     a tag string 'gauss'
%           - num :     The number of encapsulated distributions
%           - dim :     The dimension
%           - cform :   A char that indicates the covariance form
%                       - 's' : scalar form
%                       - 'd' : diagonal form
%                       - 'f' : full matrix form
%           - mu :      The mean vector(s)
%           - cvals :   The values to represent covariance(s)
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

% for argument: cform

if ~(ischar(cform) && isscalar(cform))
    error('pli_makegauss:invalidarg', 'cform should be a character.');
end

% for argument: cvals

if ~(isfloat(cvals) && isreal(cvals))
    error('pli_makegauss:invalidarg', 'cov should be a real matrix.');
end

if cform == 's'
    if ~(isscalar(cvals) || (isvector(cvals) && length(cvals) == m))
        error('pli_makegauss:invalidarg', 'Invalid size of cvals.');
    end
    if size(cvals, 2) > 1
        cvals = cvals.';
    end
    
elseif cform == 'd'
    m2 = size(cvals,2);
    if ~(ismatrix(cvals) && size(cvals,1) == d && (m2 == 1 || m2 == m))
        error('pli_makegauss:invalidarg', 'Invalid size of cvals.');
    end
           
elseif cform == 'f'
    m2 = size(cvals,3);
    if ~(size(cvals,1) == d && size(cvals,2) == d && (m2 == 1 || m2 == m))
        error('pli_makegauss:invalidarg', 'Invalid size of cvals.');
    end
    
else
    error('pli_makegauss:invalidarg', 'Invalid value of cform.');
end


        
%% make struct

G = struct( ...
    'tag', 'gauss', ...
    'num', m, ...
    'dim', double(d), ...
    'cform', cform, ...
    'mu', mu, ...
    'cvals', cvals);


