function [x, objv, exitflag] = pli_fminnewton(f, x0, opts)
%PLI_FMINNEWTON Minimizes a function using Newton-Raphson method
%
%   x = PLI_FMINNEWTON(f, x0);
%   x = PLI_FMINNEWTON(f, x0, opts);
%       
%       Minimizes the objective function f (starting from an initial
%       solution x0).
%
%       opts is an option struct, which can be generated using
%       pli_optimset. Please refer to pli_optimset for more details.
%
%   [x, objv] = PLI_FMINNEWTON(f, x0, ...);
%   [x, objv, exitflag] = PLI_FMINNEWTON(f, x0, ...);
%
%       Additionally returns the objective function value, and
%       exitflag.
%        
%       Meaning of exitflag values:
%
%           2   Pre-matured stop due to too-small change in x
%           1   Converged
%           0   Terminated without convergence (up to maxiter)
%           
%


%% argument checking

if ~isa(f, 'function_handle')
    error('pli_fminnewton:invalidarg', 'f should be a function handle.');
end

if nargin < 3
    opts = pli_optimset('newton');
end

maxiter = opts.maxiter;
tolx = opts.tolx;
tolf = opts.tolfun;
backtrk = opts.backtrk;
displevel = optim_displevel(opts.display);

eta_lb = 1.0e-12;

%% main

exitflag = 0;
t = 0;

x = x0;
[objv, g, H] = f(x);


if displevel >= 2
    fprintf('%7s  %15s  %15s  %15s  %6s\n', ...
        'Iters', 'Fval', 'Fval.ch', '1st-ord norm', 'backtrks');
    
    ppat = '%7d  %15.6g  %15.6g  %15.6g  %6d\n';    
    
    fprintf(ppat, 0, objv, nan, norm(g, inf), 0);
end


while ~exitflag && t < maxiter;
           
    t = t + 1;
    x_pre = x;
    objv_pre = objv;
    
    % update
    
    p = H \ g;
    cx = x - p;
    [cv, cg] = f(cx);
    
    btrks = 0;        
    
    if cv < objv
        x = cx;
        objv = cv;
        g = cg;                
    else    
        eta = 1.0;
        while cv >= objv && eta > eta_lb
            btrks = btrks + 1;
            eta = eta * backtrk;
            cx = x - eta * p;
            cv = f(cx);
        end
        
        if cv < objv
            x = cx;
            [objv, g, H] = f(x);
        else
            exitflag = 2;
        end
    end
    
    % determine convergence
            
    if ~exitflag        
        if abs(objv - objv_pre) < tolf && norm(x - x_pre, inf) < tolx
            exitflag = 1;
        end        
    end
    
    if displevel >= 2
        fprintf(ppat, t, objv, ...
            objv - objv_pre, norm(g, inf), btrks);
    end

end

if displevel >= 1
    fprintf('fminnewton terminated with exitflag %d\n', exitflag);
end



