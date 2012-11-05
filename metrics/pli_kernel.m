function K = pli_kernel(name, X, Y, varargin)
%PLI_KERNEL Kernel evaluation
%
%   K = PLI_KERNEL('poly', X, Y, d);
%       
%       Evaluates Polynomial kernel between the columns in X and those 
%       in Y. d is the degree of the polynomial.
%
%   K = PLI_KERNEL('gauss', X, Y, sigma);
%
%       Evaluates Gaussian kernel between the columns in X and those in Y.
%       sigma is the band-width of the kernel.
%

%% argument checking

if ~ischar(name)
    error('pli_kernel:invalidarg', ...
        'The first argument should be a kernel name.');
end

if ~(isfloat(X) && isreal(X) && ismatrix(X))
    error('pli_kernel:invalidarg', 'X should be a real matrix.');
end

if ~isempty(Y)

    if ~(isfloat(Y) && isreal(Y) && ismatrix(Y))
        error('pli_kernel:invalidarg', 'Y should be a real matrix.');
    end

    if size(X,1) ~= size(Y,1)
        error('pli_kernel:invalidarg', ...
            'The dimensions of X and Y are inconsistent.');
    end
end

%% main

switch lower(name)
    case 'poly'
        K = poly_kernel(X, Y, varargin{:});
        
    case 'gauss'
        K = gauss_kernel(X, Y, varargin{:});
        
    otherwise
        error('pli_kernel:invalidarg', ...
            'Unknown kernel name %s', name);
end

%% Core

function K = poly_kernel(X, Y, d)

if isempty(Y)
    K = X' * X;
else
    K = X' * Y;
end

K = (K + 1).^d;
    

function K = gauss_kernel(X, Y, sigma)

if isempty(X)
    D = pli_pw_euclidean(X, [], 'sq');
else
    D = pli_pw_euclidean(X, Y, 'sq');
end

K = exp((-1 / sigma^2) * D);


