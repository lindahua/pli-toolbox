function [x, objv] = pli_qpsolve(H, f, A, b, Aeq, beq, lb, ub, solver, opts)
%PLI_QPSOLVE Solve a QP problem using specified solver
%
%   [x, objv] = PLI_QPSOLVE(H, f, A, b, Aeq, beq, lb, ub, solver);
%

if ischar(solver)
    switch solver
        case 'ip'
            S = optimset( ...
                'Algorithm', 'interior-point-convex', ...
                'Display', 'off');
            if ~isempty(opts)
                S = optimset(S, opts);
            end
            [x, objv] = quadprog(H, f, A, b, Aeq, beq, lb, ub, [], S);            
            
        case 'gurobi'
            if ~exist('gurobi', 'file')
                error('pli_qpsolve:invalidarg', ...
                    'Gurobi solver is NOT found.');           
            end
            
            model.Q = 0.5 * H;
            model.obj = f;
            d = length(f);
            
            if isempty(Aeq)
                model.A = A;
                model.rhs = b;
                model.sense = '<';
                
            elseif isempty(A)
                model.A = Aeq;
                model.rhs = beq;
                model.sense = '=';
                
            else
                m1 = size(A, 1);
                m2 = size(Aeq, 1);
                
                model.A = [A; Aeq];
                model.rhs = [b; beq];
                model.sense = [repmat('>', m1, 1); repmat('=', m2, 1)];
            end
            
            if isempty(lb)
                model.lb = -inf(d, 1);
            else
                if ~all(lb == 0)
                    model.lb = lb;
                end
            end
            
            if ~isempty(ub)
                model.ub = ub;
            end
            
            if isempty(opts)
                opts.outputflag = 0;
            else
                if ~isfield(opts, 'outputflag')
                    opts.outputflag = 0;
                end
            end
               
            ret = gurobi(model, opts);
            x = ret.x;
            objv = ret.objval;
            
        otherwise
            error('pli_qpsolve:invalidarg', ...
                'Unknown solver name %s', solver);            
    end
    
elseif isa(solver, 'function_handle')

    if ~isempty(opts)
        warning('pli_qpsolve:invalidarg', ...
            'opts is ignored with solver is input as a function handle.');
    end
    
    x = solver(H, f, A, b, Aeq, beq, lb, ub);
        
else
    error('pli_qpsolve:invalidarg', ...
        'The solver is invalid.');
end
    


