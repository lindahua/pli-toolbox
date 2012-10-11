function [v, g] = pli_linsvm_objv(X, y, lambda, lambda0, h, theta, bias)
%PLI_LINSVM_OBJV Objective function of Linear SVM
%
%   The objective function is defined to be
%
%         (lambda/2) * ||theta||^2
%       + (lambda0/2) * (theta0)^2
%       + sum_i loss(y_i * (theta' * x_i + bias); h).
%
%   Here, the loss function is defined as follows.
%
%   When h == 0, it is a hinge loss, as
%
%       loss(u; 0) = max(1 - u, 0);
%
%   When h > 0, it is a huber loss, as
%
%                    | 0                     (when u >= 1 + h)
%       loss(u; h) = | (1+h-u)^2 / (4h)      (when 1-h < u < 1+h)
%                    | 1 - u                 (when u <= 1 - h)
%
%   It is obvious that the huber loss degenerates to hinge loss when
%   h equals 0.
%
%
%   v = PLI_LINSVM_OBJV(X, y, lambda, lambda0, h, theta, bias)
%   v = PLI_LINSVM_OBJV(X, y, lambda, lambda0, h, theta, bias);
%
%       Evaluates the linear SVM objective as above. When bias is not
%       given, it is assumed to be fixed to zero.
%
%   [v, g] = PLI_LINSVM_OBJV(X, y, lambda, lambda0, h, theta);
%   [v, g] = PLI_LINSVM_OBJV(X, y, lambda, lambda0, h, theta, bias);
%
%       Additionally evaluates the gradient w.r.t. [theta; bias] 
%       or w.r.t. theta (depending on whether bias is given).
%
%   
%   This function is supposed to be called within an optimization 
%   procedure. For efficiency, no argument checking is performed.
%

%% objective value

if nargin < 7
    use_bias = 0;
else
    use_bias = 1;
end

n = size(X, 2);

rv = (0.5 * lambda) * norm(theta)^2;
if use_bias && lambda0 > 0
     rv = rv + (0.5 * lambda0) * (bias^2);
end

u = y .* (theta' * X + bias);
sv = find(u < 1 + h);

% evaluate objective

if isempty(sv)
    tloss = 0;
else
    u_sv = u(sv);
    
    if h == 0
        tloss = numel(u_sv) - sum(u_sv);        
    else
        u_sv = u(sv);
        in_q = (u_sv > 1 - h);
        il = find(~in_q);
        iq = find(in_q);
        
        if isempty(il)
            tloss_l = 0;
        else
            tloss_l = numel(il) - sum(u_sv(il));
        end
        
        if isempty(iq)
            tloss_q = 0;
        else
            tloss_q = sum((1 + h - u_sv(iq)).^2) / (4*h);
        end
        
        tloss = tloss_l + tloss_q;
    end
end

v = rv + tloss / n;

%% gradient

if nargout >= 2
    
    g = theta * lambda;
    
    if ~isempty(sv)

        X_sv = X(:, sv);
        y_sv = y(sv);
        
        ya = y_sv;
        if h > 0 && ~isempty(iq)
            ya(iq) = ((1+h) - u_sv(iq)) * (1/(2*h));
        end
        
        g = g - (X_sv * ya') * (1/n);
    end
    
    if use_bias
       
        g0 = bias * lambda0;
       
        if ~isempty(sv)
            g0 = g0 - sum(ya) * (1/n);
        end
        
        g = [g; g0];
    end
            
end


