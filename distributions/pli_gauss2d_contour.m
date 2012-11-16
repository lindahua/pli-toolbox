function pli_gauss2d_contour(G, r, n, varargin)
%PLI_GAUSS2D_CONTOUR Draws a Gaussian elliptic contour
%
%   PLI_GAUSS2D_CONTOUR(G);
%       Draws a standard Gaussian contour (the points whose Mahalanobis
%       distance to the center is 1).
%
%       When G.num > 1, it draws multiple contours, each for one
%       distribution in G.
%
%   PLI_GAUSS2D_CONTOUR(G, r);
%       Draws a standard Gaussian contour, where the Malalanobis distance
%       to the center is r.
%
%   PLI_GAUSS2D_CONTOUR(G, r, n);
%       n is the sample density (i.e. the number of samples lying 
%       on the contour).
%
%       When n is omitted, it is set to 500 by default.
%
%   PLI_GAUSS2D_CONTOUR(G, r, n, ...);
%
%       One can further specify the line specification through ensuing
%       arguments.
%

%% argument checking

if ~(isstruct(G) && strcmp(G.tag, 'gauss') && G.dim == 2)
    error('pli_gauss2d_contour:invalidarg', ...
        'G should be a Gaussian struct with G.dim == 2.');
end

if nargin < 2
    r = 1;
else
    if ~(isscalar(r) && isreal(r) && isfloat(r) && r > 0)
        error('pli_gauss2d_contour:invalidarg', ...
            'r should be a positive real scalar.');
    end
end

if nargin < 3
    n = 500;
else
    if ~(isscalar(n) && isreal(n) && n == fix(n) && n > 1)
        error('pli_gauss2d_contour:invalidarg', ...
            'n should be a positive integer scalar with n > 1.');
    end
end

%% main

t = linspace(0, 2*pi, n);
x0 = r * cos(t);
y0 = r * sin(t);

m = G.num;

if m == 1
    [xt, yt] = transform_single(G.mu, G.cvals, G.cform, x0, y0);
    plot(xt, yt, varargin{:});
else
    cf = G.cform;
    for k = 1 : m        
        [mu_k, cov_k] = get_single(G, k);
        [xt, yt] = transform_single(mu_k, cov_k, cf, x0, y0);
        
        if k > 1
            hold on;
        end
        plot(xt, yt, varargin{:});
    end
end


function [xt, yt] = transform_single(mu, cov, cf, x, y)

switch cf
    case 's'
        s = sqrt(cov);
        xt = s * x;
        yt = s * y;
    case 'd'
        s = sqrt(cov);
        xt = s(1) * x;
        yt = s(2) * y;
    case 'f'
        % 2D Cholesky decompostion
        c11 = cov(1, 1);
        c12 = cov(1, 2);
        c22 = cov(2, 2);
        
        % L = [a 0; b c]
        
        a = sqrt(c11);
        b = c12 / a;
        c = sqrt(c22 - b^2);
        
        % apply transform
        xt = a * x;
        yt = b * x + c * y;
end

if ~isequal(mu, 0)
    xt = xt + mu(1);
    yt = yt + mu(2);
end


function [mu, cov] = get_single(G, i)
% get the mean and covariance of a particular Gauss


if size(G.mu, 2) == 1
    mu = G.mu;
else
    mu = G.mu(:, i);
end

G_cvals = G.cvals;

switch G.cform
    case 's'
        if isscalar(G_cvals)
            cov = G_cvals;
        else
            cov = G_cvals(i);
        end
    case 'd'
        if size(G_cvals, 2) == 1
            cov = G_cvals;
        else
            cov = G_cvals(:, i);
        end
    case 'f'
        if ismatrix(G_cvals)
            cov = G_cvals;
        else
            cov = G_cvals(:,:,i);
        end
end

