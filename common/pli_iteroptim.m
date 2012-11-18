function [sol, objv, t, converged] = pli_iteroptim(op, objfun, updatefun, sol, varargin)
%PLI_ITEROPTIM Optimize via iterative updating
%
%   sol = PLI_ITEROPTIM('maximize', objfun, updatefun, sol, ...);
%   sol = PLI_ITEROPTIM('minimize', objfun, updatefun, sol, ...);
%
%       Optimizes a given solution through iterative updating.
%
%   sol, objv, t, converged = PLI_ITEROPTIM( ... );
%       
%   Arguments
%   ---------
%   - objfun :      A function to evaluate objective w.r.t. to a solution.
%
%   - updatefun :   The function to update the solution. 
%   
%   - sol :         The solution to be updated.
%
%   Returns
%   -------
%   - sol :         The optimized solution.
%
%   - objv :        The objective value at final step.
%
%   - t :           The number of elapsed iterations.
%
%   - converged :   Whether the procedure converges.
%
%   One can specify other options in the form of name/value pairs to 
%   control the procedure.
%
%   - maxiter :     The maximum number of iterations. (default = 200)
%
%   - tolfun :      The tolerance of objective function changes at
%                   convergence. (default = 1.0e-8)
%
%   - nupdates :    The number of updates at each iteration. (default = 1).
%   
%   - display :     'off' | 'final' | {'iter'}.
%

%% argument checking

if strcmpi(op, 'maximize')
    optim_dir = 1;
elseif strcmpi(op, 'minimize')
    optim_dir = -1;
else
    error('pli_iteroptim:invalidarg', ...
        'The first argument is invalid.');
end

if ~isa(objfun, 'function_handle')
    error('pli_iteroptim:invalidarg', ...
        'objfun must be a function handle.');
end

if ~isa(updatefun, 'function_handle')
    error('pli_iteroptim:invalidarg', ...
        'updatefun must be a function handle.');
end

[maxiter, tolfun, nupdates, displevel] = get_opts(varargin);

%% main

t = 0;
converged = false;

objv = nan;

if displevel >= 2
    fprintf('#Iter         obj.value        obj.change\n');
    fprintf('=============================================\n');
end

while ~converged && t < maxiter
    t = t + 1;
    pre_objv = objv;
    
    if displevel >= 2
        fprintf('%5d ', t);
    end
    
    % update solution
    
    if nupdates == 1
        sol = updatefun(sol);
    else
        for j = 1 : nupdates
            sol = updatefun(sol);
        end
    end
    
    % re-evaluate objective function
    
    objv = objfun(sol);
    
    if t > 1
        ch = objv - pre_objv;
        
        % determine convergence
        
        if abs(ch) < tolfun
            converged = true;
        end
    end
    
    if displevel >= 2
        if t == 1
            fprintf('\t%15g\n', objv);
        else
            fprintf('\t%15g   %15g\n', objv, ch);
        end        
    end
    
    % diagnosis
    
    if t > 1
        if ch * optim_dir < -tolfun
            warning('iter_optim:objvchange', ...
                'Unexpected objective function changes: %g', ch);
        end
    end
    
end

if displevel >= 1
    if converged
        fprintf('Optimization converged at t = %d\n', t);
    else
        fprintf('Optimization terminated without convergence at t = %d\n', t);
    end
    fprintf('\n');
end


%% auxiliary functions

function [maxiter, tolfun, nupdates, displevel] = get_opts(oplist)

S = struct( ...
    'maxiter', 200, 'tolfun', 1.0e-8, 'nupdates', 1, ...
    'display', 'iter');

if ~isempty(oplist)
    S = pli_parseopts(S, oplist);
end

maxiter = S.maxiter;
tolfun = S.tolfun;
nupdates = S.nupdates;

switch S.display
    case 'off'
        displevel = 0;
    case 'final'
        displevel = 1;
    case 'iter'
        displevel = 2;
    otherwise
        error('pli_iteroptim:invalidarg', ...
            'Invalid value for the display option.');
end




        