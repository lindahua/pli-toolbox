function [theta, theta0] = pli_pegasos(X, y, lambda, lambda0, T, s0)
%PLI_PEGASOS Linear Support vector machine using Pegasos algorithm
%
%   The linear SVM problem is formulated as follows
%
%       minimize (lambda/2) * ||w||^2 + (lambda0/2) * (w0)^2
%               (1/n) * sum_i Loss_i
%
%   Here,
%
%       Loss_i = max(1 - y_i (w' * x_i + w0), 0).
%
%
%   [w, w0] = PLI_DIRECTSVM(X, y, lambda, T, aug);
%   [w, w0] = PLI_DIRECTSVM(X, y, lambda, T, aug, s0);
%
%       Solves the coefficient w and offset w0 using for a linear
%       SVM for using Pegasos algorithm.
%
%   Arguments
%   ---------
%   X :         The matrix of sample features, size [d n], where
%               d is the sample dimension (#features), and n is 
%               the number of observed samples.
%
%   y :         The response vector of length n. Each entry of y
%               should be either 1 or -1.
%
%   lambda :    The regularization coefficient for theta.
%
%   lambda0 :   The regularization coefficient for theta0.
%
%   T :         The number of iterations. (Each iteration uses one
%               sample for stochastic gradient update).
%
%   s0 :        The initial guess of the solution, when should be 
%               a (d+1)-dimensional vector in the form of [w; w0].
%               By default, it is set to zeros(d+1, 1);
%
%
%   Returns
%   -------
%   w :     The coefficient vector.
%
%   w0 :    The bias (offset) value.
%

%% argument checking

if ~(ismatrix(X) && isfloat(X) && isreal(X))
    error('pli_pegasos:invalidarg', 'X should be a real matrix.');
end
[d, n] = size(X);

if ~(isvector(y) && isfloat(y) && isreal(y) && numel(y) == n)
    error('pli_pegasos:invalidarg', ...
        'y should be a real vector of length n.');
end

if ~(isfloat(lambda) && isreal(lambda) && isscalar(lambda) && lambda > 0)
    error('pli_pegasos:invalidarg', ...
        'lambda should be a positive real scalar.');
end
lambda = double(lambda);

if ~(isfloat(lambda0) && isreal(lambda0) && isscalar(lambda0) && lambda0 > 0)
    error('pli_pegasos:invalidarg', ...
        'lambda0 should be a positive real scalar.');
end
lambda0 = double(lambda0);

if ~(isscalar(T) && isreal(T) && T >= 1)
    error('pli_pegasos:invalidarg', ...
        'T should be a positive integer scalar.');        
end

if nargin < 6 || isempty(s0)
    s0 = [];
else
    if ~(isfloat(s0) && isreal(s0) && isequal(size(s0), [d+1, 1]))
        error('pli_pegasos:invalidarg', ...
            's0 should be a real vector of size [d+1, 1].'); 
    end
end

%% main

if ~isa(X, 'double'); X = double(X); end
if ~isa(y, 'double'); y = double(y); end


% initialize solution

if isempty(s0)
    init_w = zeros(d, 1);
    init_w0 = 0;
else
    init_w = s0(1:d);
    init_w0 = s0(d+1);
end

% main loop

inds = randi(n, [1 T]);

aug = sqrt(lambda / lambda0);

[theta, theta0] = pegasos_cimp(X, y, int32(inds)-1, init_w, init_w0, lambda, aug, ...
    int32(0));

if aug > 0
    theta0 = theta0 * 10;
end


function [w, w0] = ref_pegasos(X, y, inds, w, w0, lambda, aug) %#ok<DEFNU>
%A reference (pure-matlab) implmentation for debugging

disp('Ref');

T = length(inds);

for t = 1 : T
    
    % fetch a sample
           
    i = inds(t);
    xt = X(:, i);
    yt = y(i);   
    
    % evaluate predictors
    ut = w' * xt + w0 * aug;
    
    % set learning rate
    eta = 1 / (lambda * t); 
    
    % update w
    rp = (1 - eta * lambda);    
    w = rp * w;
    w0 = rp * w0;
    
    if yt * ut < 1
        w = w + eta * (yt * xt);
        w0 = w0 + eta * (yt * aug);
    end
    
    % rescale w (projection)
    w_sca = 1 / (sqrt(lambda) * norm([w; w0]));
    if w_sca < 1
        w = w * w_sca;
        w0 = w0 * w_sca;
    end    
end


