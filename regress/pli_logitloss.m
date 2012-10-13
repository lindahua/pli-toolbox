function [v, g, H] = pli_logitloss(theta, X, y, w)
%PLI_LOGITLOSS Evaluates the logit-loss
%
%   Given a sample x and response y, the logit loss w.r.t.
%   to parameter theta and theta0 is defined to be
%
%       L(x, y) := y * log(p) + (1 - y) * log(1 - p)
%
%   Here, p := 1 / (1 + exp(-(theta * x + theta0)))
%
%   This is the loss function used in logistic regression.
%
%   
%   v = PLI_LOGITLOSS(theta, X, y);
%   v = PLI_LOGITLOSS(theta, X, y, w);
%   v = PLI_LOGITLOSS([theta; theta0], X, y);
%   v = PLI_LOGITLOSS([theta; theta0], X, y, w);
%       
%       Evaluates the total logit-loss over all given samples. 
%
%       The first argument is the parameter, which can be a d-by-1
%       vector theta (in this case, bias term is fixed to zero), or
%       a (d+1)-by-1 vector in the form of [theta; theta0].
%
%       Suppose there are n samples of dimension d. X should be 
%       a d-by-n matrix, and y should be a 1-by-n row vector.
%
%       w is the sample weights. If omitted, all samples have the 
%       unit weight. w should be a column vector of length n.
%
%   [v, g] = PLI_LOGITLOSS( ... );
%
%       Additionally evaluates the gradient w.r.t. to theta or 
%       [theta; theta0].
%
%   [v, g, H] = PLI_LOGITLOSS( ... );
%
%       Additionally evaluates the Hessian matrix.
%
%   
%   This function is supposed to be used within an optimization problem.
%   For efficiency, no argument checking is performed.
%

%% main

if nargin < 4
    w = [];
end

% make prediction

d = size(X, 1);
ds = size(theta, 1);

if ds == d    
    use_bias = 0;
    u = theta' * X;
else
    use_bias = 1;
    
    theta0 = theta(d+1);
    theta = theta(1:d);
    
    u = theta' * X + theta0;
end

% evaluate objective

loss = y .* log1p_exp(-u) + (1 - y) .* log1p_exp(u);

if isempty(w)
    v = sum(loss);
else
    if size(w, 2) > 1
        w = w.';
    end
    v = loss * w;
end

% evaluate gradient

if nargout >= 2
    
    p = 1 ./ (1 + exp(-u));
    pmy = p - y;
    
    if isempty(w)
        g = X * pmy';
    else
        g = X * (pmy' .* w);
    end    
    
    if use_bias
        if isempty(w)
            g0 = sum(pmy);
        else
            g0 = pmy * w;
        end
        
        g = [g; g0];
    end
end

% evaluate Hessian

if nargout >= 3
    
    rho = p .* (1 - p);
    ri = find(rho > 0);
    
    if ~isempty(ri)
        Xr = X(:, ri);
        if isempty(w)
            rw = rho(ri);
        else
            rw = rho(ri) .* w(ri);
        end
        
        G = Xr * bsxfun(@times, Xr, rw)';
        G = 0.5 * (G + G');
    end
    
    if use_bias
        h = Xr * rw';
        H = [G h; h' sum(rw)];
    else
        H = G;
    end
    
end



