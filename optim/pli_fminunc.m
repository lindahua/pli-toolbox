function [x, fval, exitflag] = pli_fminunc(f, x0, opts)
%PLI_FMINUNC Unconstrained nonlinear optimization
%
%   [x, fval] = PLI_FMINUNC(f, x0);
%   [x, fval] = PLI_FMINUNC(f, x0, opts);
%
%       Tries to find a local minima of a function f, starting from x0.
%       By default, the BFGS method is used.
%
%       opts is an option to control the optimization procedure. 
%       See PLI_OPTIMSET for more information about the options.
%
%       The function handle f may be used as follows:
%
%           v = f(x);       % f outputs the function value
%           [v, g] = f(x);  % f outputs the function value and gradient
%
%       Not each call to f requests the gradient. It is advisable for 
%       f to determine whether to calculate the gradient according to
%       the number of outputs.
%
%       When the method 'newton' is used, f may also produce a Hessian
%       matrix as follows,
%
%           [v, g, H] = f(x);
%
%
%   [x, fval, exitflag] = PLI_FMINUNC(f, x0, ...);
%
%       Additionally returns a flag that characterizes the condition
%       under which the optimization procedure terminated.
%
%       It can take either of the following values (the meaings of
%       these values are consistent with that for the fminunc function
%       in MATLAB's optimization toolbox).
%       
%           1:  converged (gradient is small enough)
%           2:  change in x is too small
%           3:  change in objective function is too small
%           5:  failed to decrease the objective along search direction.
%           0:  maximum number of iterations reached before convergence.  
%          -3:  the objective is not bounded. 
%           
%   Remarks
%   -------
%       - This function does not check the validity of opts. One may use
%         PLI_OPTIMSET to construct the opts, which guarantees the 
%         validity of the option struct.
%
%       - When method 'steepest' is used. The second output argument
%         of f may or may not be the gradient at x, which can be any
%         search direction (one can implement f such that it computes
%         a good direction within f and returns it as the second
%         output).
%

%% argument checking

if ~isa(f, 'function_handle')
    error('pli_fminunc:invalidarg', 'f should be a function handle.');
end

if ~(isfloat(x0) && isreal(x0) && isvector(x0))
    error('pli_fminunc:invalidarg', 'x0 should be a real vector.');    
end
if size(x0, 2) > 1
    x0 = x0.';
end
d = size(x0, 1);

if nargin < 3
    opts = pli_optimset();
end

%% main

% extract options

maxiter = opts.maxiter;
tolgrad = opts.tolgrad;
tolfun = opts.tolfun;
tolx = opts.tolx;
displevel = optim_displevel(opts.display);

use_hess = false;

switch opts.method
    case 'steepdesc'
        mcode = 0;
    case 'bfgs'
        mcode = 1;            
    case 'cg'
        mcode = 2;      
        
        switch opts.cgscheme
            case 'f-r'
                cg_code = 1;
            case 'p-r'
                cg_code = 2;
            otherwise
                error('pli_fminunc:invalidarg', ...
                    'Invalid value for cg_scheme: %s', opt.cg_scheme);
        end
        
    case 'newton'
        mcode = 3;
        use_hess = true;
    otherwise
        error('pli_fminunc:invalidarg', ...
            'Invalid method: %s', opts.method);
end

% prepare for mainloop

x = x0;
exitflag = 0;
t = 0;

c1 = 1.0e-4;  % the coefficient for Armijo rule
step_lb = 1.0e-12;
btrk_r = 0.618;

if ~use_hess
    [fval, g] = f(x);
else
    [fval, g, H] = f(x);
end

if mcode == 1       % bfgs
    H = eye(d);
elseif mcode == 2   % cg
    s = 0;    
    cg_beta = 0;
end
    
 
% display head line

if displevel >= 2
    fprintf('%7s  %15s  %15s  %15s  %15s  %9s\n', ...
        'Iters', 'fval', '1st-ord norm', 'fval.change', 'x.change', '#backtrks');
    
    ppat = '%7d  %15.6g  %15.6g  %15.6g  %15.6g  %9d\n';
    
    fprintf(ppat, 0, fval, norm(g, inf), nan, nan, 0);
end

% main loop

while ~exitflag && t < maxiter    
    
    t = t + 1;
    x_pre = x;
    g_pre = g;
    fval_pre = fval;
    
    nbtrks = 0;
       
    % find search direction (using specified method)
    
    if mcode < 2
        if mcode == 0
            s = g;
        else
            s = H \ g;
        end
    else
        if mcode == 2
            s = g + cg_beta * s;
        else
            s = H \ g;
        end
    end
    
    % line search
    
    % go with full step and evaluate
    
    a = 1.0;  % step size
    cx = x - s;    
    
    if ~use_hess
        [cv, cg] = f(cx);
    else
        [cv, cg, cH] = f(cx);
    end
            
    sg = s' * g;
    if cv <= fval && cv <= fval + c1 * sg
        % if satisfy Armijo rule, use current update
        
        x = cx;
        fval = cv;
        g = cg;
        
        if use_hess
            H = cH;
        end
        
        step_found = true;
    else
        % otherwise, perform back-tracking
        
        a = 1.0;
        step_found = false;
        while ~step_found && a > step_lb
            nbtrks = nbtrks + 1;
            a = a * btrk_r;
            
            cx = x - s * a;            
            cv = f(cx);
                        
            if cv <= fval && cv <= fval + c1 * a * sg
                % found a good step size
                
                x = cx;
                
                if ~use_hess
                    [fval, g] = f(cx);
                else
                    [fval, g, H] = f(cx);
                end
                
                step_found = true;
            end
        end
    end
    
    % check termination condition
    
    xchange = [];
    
    if ~step_found
        exitflag = 5;
    
    elseif isinf(fval)
        exitflag = -3;
        
    elseif norm(g, inf) < tolgrad
        exitflag = 1;
        
    elseif abs(fval - fval_pre) < tolfun
        exitflag = 3;
        
    else
        xchange = a * norm(s, inf);
        if xchange < tolx
            exitflag = 2;
        end
    end
        
    % update relevant states
    
    if ~exitflag
        
        if mcode == 1 % bfgs
            % update H
            
            dx = x - x_pre;
            dg = g - g_pre;
            
            Hdx = H * dx;
            H = H + (dg * dg') * (1 / (dg' * dx)) - (Hdx * Hdx') * (1 / (dx' * Hdx));
            
        elseif mcode == 2  % cg
            
            if cg_code == 1 % Fletcher-Reeves                
                cg_beta = (g' * g) / (g_pre' * g_pre);                
            else % Polak-Ribiere                
                cg_beta = (g' * (g - g_pre)) / (g_pre' * g_pre);                
            end

        end    
    end
    
    % print iteration information
    
    if displevel >= 2
        if isempty(xchange)
            xchange = norm(x - x_pre, inf);
        end
            
        fprintf(ppat, t, fval, ...
            norm(g, inf), fval - fval_pre, xchange, nbtrks);
    end
        
end

% print final information

if displevel >= 1
    msg = optim_exitmsg(exitflag);
    fprintf('Terminated with %d iterations: %s.\n', t, msg);
end


