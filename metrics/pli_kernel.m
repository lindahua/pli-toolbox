function K = pli_kernel(name, X, Y, varargin)
%PLI_KERNEL Kernel evaluation
%
%   K = PLI_KERNEL(kername, X, [], ...);
%   K = PLI_KERNEL(kername, X, Y, ...);
%
%       This is the general syntax of using this function.
%
%       Evaluates the specified kernel function between columns in X and
%       those in Y. Suppose X and Y respectively have m and n columns,
%       then the resultant matrix K will be a matrix of size m-by-n.
%
%       Setting Y to empty means to computer pairwise kernel values 
%       between columns in X. 
%
%
%   Specific kernels supported by this function and their usage:
%
%   K = PLI_KERNEL('lin', X, Y);
%
%       Linear kernel: k(x, y) = x' * y;
%
%
%   K = PLI_KERNEL('poly', X, Y, d, a);
%       
%       Polynomial kernel: k(x, y; d, a) = (x' * y + a)^d
%       Default params: d = 2, a = 1.
%
%
%   K = PLI_KERNEL('gauss', X, Y, sigma);
%   K = PLI_KERNEL('rbf', X, Y, sigma);
%
%       Gaussian RBG kernel: 
%           k(x, y; sigma) = exp(- (1/2) * (x - y)^2 / sigma^2)
%
%       Default params: sigma = 1.
%
%
%   K = PLI_KERNEL('tanh', X, Y, a, b);
%   K = PLI_KERNEL('mlp', X, Y, a, b);
%
%       Multi-layer perceptron kernel:
%           K(x, y; a, b) = tanh(a * x' * y + b);
%       Default params: a = 1, b = -1.           
%
%
%   K = PLI_KERNEL(kfun, X, Y, ...);
%
%       User-defined kernel function. Here, kfun is a function-handle.
%
%       kfun(X, Y, ...) should return an m-by-n matrix, suppose X and Y
%       respectively have m and n columns.
%

%% argument checking

if ischar(name)
    use_preset = 1;
elseif isa(name, 'function_handle')
    use_preset = 0;
    kfun = name;    
else    
    error('pli_kernel:invalidarg', ...
        'The first argument should be a kernel name or a function handle.');
end

if ~(isfloat(X) && isreal(X) && ismatrix(X))
    error('pli_kernel:invalidarg', 'X should be a real matrix.');
end

if nargin < 3
    Y = [];
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

if use_preset
    switch lower(name)
        case 'lin'
            K = lin_kernel(X, Y, varargin{:});
            
        case 'poly'
            K = poly_kernel(X, Y, varargin{:});
            
        case {'rbf', 'gauss'}
            K = rbf_kernel(X, Y, varargin{:});
            
        case {'mlp', 'tanh'}
            K = mlp_kernel(X, Y, varargin{:});
            
        otherwise
            error('pli_kernel:invalidarg', ...
                'Unknown kernel name %s', name);
    end
else
    if isempty(Y)
        Y = X;
    end
    
    K = kfun(X, Y, varargin{:}); 
end


%% kernel functions

function K = lin_kernel(X, Y)

if isempty(Y)
    K = X' * X;
else
    K = X' * Y;
end

function K = poly_kernel(X, Y, d, a)

if nargin < 3
    d = 2;
end

if nargin < 4
    a = 1;
end

if isempty(Y)
    K = X' * X;
else
    K = X' * Y;
end

if a ~= 0
    K = K + a;
end

if d == 2
    K = K .* K;
elseif d ~= 1
    K = K .^ d;
end
    

function K = rbf_kernel(X, Y, sigma)

if nargin < 3
    sigma = 1;
end

D = pli_pw_euclidean(X, Y, 'sq');
K = exp((-0.5 / sigma^2) * D);


function K = mlp_kernel(X, Y, a, b)

if nargin < 3; a = 1; end
if nargin < 4; b = 1; end

if isempty(Y)
    K = X' * X;
else
    K = X' * Y;
end

if a ~= 1
    K = K * a;
end

K = tanh(K + b);

