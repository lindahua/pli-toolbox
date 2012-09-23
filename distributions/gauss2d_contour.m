function h = gauss2d_contour(G, r, n, varargin)
%GAUSS2D_CONTOUR Draws a Gaussian elliptic contour
%
%   gauss2d_contour(G);
%       Draws a standard Gaussian contour (the points whose Mahalanobis
%       distance to the center is 1).
%
%   gauss2d_contour(G, r);
%       Draws a standard Gaussian contour, where the Malalanobis distance
%       to the center is r.
%
%   gauss2d_contour(G, r, n);
%       n is the sample density (i.e. the number of samples lying 
%       on the contour).
%
%       When n is omitted, it is set to 500 by default.
%
%   gauss2d_contour(G, r, n, ...);
%
%       One can further specify the line specification through ensuing
%       arguments.
%

%% argument checking

if ~(isstruct(G) && strcmp(G.tag, 'gauss') && G.num == 1)
    error('gauss2d_contour:invalidarg', ...
        'G should be a Gaussian struct with G.num == 1.');
end

if nargin < 2
    r = 1;
else
    if ~(isscalar(r) && isreal(r) && isfloat(r) && r > 0)
        error('gauss2d_contour:invalidarg', ...
            'r should be a positive real scalar.');
    end
end

if nargin < 3
    n = 500;
else
    if ~(isscalar(n) && isreal(n) && n == fix(n) && n > 1)
        error('gauss2d_contour:invalidarg', ...
            'n should be a positive integer scalar with n > 1.');
    end
end

%% main

t = linspace(0, 2*pi, n);

Z = r * [cos(t); sin(t)];

cov = G.cov;
if isscalar(cov)
    X = sqrt(cov) * Z;
elseif size(cov, 2) == 1
    X = bsxfun(@times, sqrt(cov), Z);
else
    L = chol(cov, 'lower');
    X = L * Z;
end

if ~isequal(G.mu, 0)
    X = bsxfun(@plus, X, G.mu);
end

if nargout == 0
    plot(X(1,:), X(2,:), varargin{:});
else
    h = plot(X(1,:), X(2,:), varargin{:});
end


