function [w, w0, objv] = pli_directsvm(X, y, c, h, s0, solver)
%PLI_DIRECTSVM Solve SVM by directly optimizing the primal function
%
%   The objective function is given by 
%
%       (1/2) ||w||^2 + r0 * w0^2 + 
%       sum_i c * huber_loss(y_i * (w' * x_i + w0))
%
%   Here, 
%                         0                     (if u >= 1 + h)
%       huber_loss(u) =   (1 + h - u)^2 / (4h)  (if |1-u| < h)
%                         1 - u                 (if u <= 1 - h)
%
%   The coefficient r0 is often set to a very small value, which 
%   is 1e-6 in this function.
%
%
%   [w, w0] = PLI_DIRECTSVM(X, y, c);
%   [w, w0] = PLI_DIRECTSVM(X, y, c, h);
%   [w, w0] = PLI_DIRECTSVM(X, y, c, h, s0);
%   [w, w0] = PLI_DIRECTSVM(X, y, c, h, s0, solver);
%
%       Learns SVM from given data by directly optimizing the primal
%       objective function (in uncontrained form with huber loss).
%
%   [w, w0, objv] = PLI_DIRECTSVM( ... );
%
%       Additionally returns the objective value.
%
%   Arguments
%   ---------
%   - X :       The input feature matrix, of size [d n]
%
%   - y :       The response vector, of length n.
%
%   - c :       The coefficient of loss, which should be a scalar.
%
%   - h :       The (half) transition width of the huber loss.
%               (default = 0.1)
%
%   - s0 :      The initial guess of solution, which should be a 
%               (d+1)-dimensional vector. By default, it is set
%               to be a zero vector.
%
%   - solver :  The solver function handle, which will be used as
%
%                   [x, objv] = solver(f, x0);
%           
%               By default, @pli_fminbfgs will be used.
%
%
%   This approach, while apparently simple, can be very efficient for
%   cases where d << n, which is often the case in computer vision 
%   applications.
%

%% argument checking

if ~(isfloat(X) && isreal(X) && ismatrix(X))
    error('pli_directsvm:invalidarg', ...
        'X should be a real matrix.');
end
[d, n] = size(X);

if ~(isfloat(y) && isreal(y) && isvector(y) && length(y) == n)
    error('pli_directsvm:invalidarg', ...
        'y should be a real vector of length n.');
end
if size(y, 1) > 1; y = y.'; end

if ~(isfloat(c) && isreal(c) && isscalar(c) && c > 0)
    error('pli_directsvm:invalidarg', ...
        'c should be either a positive real scalar.');
end

if nargin < 4
    h = 0.1;
else
    if ~(isfloat(h) && isreal(h) && isscalar(h) && h > 0 && h < 1)
        error('pli_directsvm:invalidarg', ...
            'h should be a positive real value in (0, 1).');
    end
end

if nargin < 5 || isempty(s0)
    s0 = zeros(d+1, 1);
else
    if ~(isfloat(s0) && isreal(s0) && isequal(size(s0), [d+1 1]))
        error('pli_directsvm:invalidarg', ...
            's0 should be a real vector of size [d+1 1].');
    end
end

if nargin < 6
    solver = @pli_fminbfgs;
else
    if ~isa(solver, 'function_handle')
        error('pli_directsvm:invalidarg', ...
            'solver should be a function handle.');
    end
end

r0 = 1.0e-6;

%% main

[sol, objv] = solver(@objfun, s0);

w = sol(1:d);
w0 = sol(d+1);


%% objective function

    function [objv, g] = objfun(s)
        
        t = s(1:d);
        t0 = s(d+1);
        
        rv = 0.5 * norm(t)^2 + (0.5 * r0) * (t0^2);
        
        u = y .* (t' * X + t0);
        sv = find(u < 1 + h);
        
        % evaluate objective
        
        if isempty(sv)
            tloss = 0;
        else
            u_sv = u(sv);
            in_q = (u_sv > 1 - h);
            il = find(~in_q);
            iq = find(in_q);
            
            if isempty(il)
                tloss_l = 0;
            else
                tloss_l = sum(1 - u_sv(il));
            end
            
            if isempty(iq)
                tloss_q = 0;
            else
                tloss_q = sum((1 + h - u_sv(iq)).^2) / (4*h);
            end
            
            tloss = c * (tloss_l + tloss_q); 
        end
            
        objv = rv + tloss;
        
        % evaluate gradient (upon request)
        
        if nargout >= 2
            
            if isempty(sv)
                g = [t; t0 * r0];
            
            else
                X_sv = X(:, sv);
                y_sv = y(sv);
                
                a = (max(u_sv, 1-h) - (1+h)) * (1/(2*h));
                ya = y_sv .* a;
                
                g = [(X_sv * ya') * c + t; sum(ya) * c + t0 * r0];
            end
        end

    end

end


