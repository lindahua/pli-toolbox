function opts = pli_optimset(varargin)
%PLI_OPTIMSET Sets the optimization options
%
%   opts = PLI_OPTIMSET();
%
%       Returns the default option struct.
%
%   opts = PLI_OPTIMSET('name1', val1, 'name2', vals, ...);
%
%       Returns an option struct, whose values are overrided by
%       the values provided in the form of name/value list.
%
%       The options which do not appear in the input list still
%       use the default values.
%
%   opts = PLI_OPTIMSET(new_opts);
%
%       Returns an option struct, whose values are overrided by
%       the values provided in new_opts.
%
%   List of supported options:
%
%   - method :      The name of the method to use:
%                   - 'steepdesc':  Steepest descent
%                   - 'bfgs':       Quasi-newton using BFGS update
%                   - 'cg':         Nonlinear conjugate gradient
%                   - 'newton':     Newton-Raphson
%                   default = 'bfgs'.
%
%   - cgscheme :    The scheme to calculate the beta coefficient 
%                   in Conjugate gradient. (only for method 'cg')
%                   - 'f-r':    Fletcher-Reeves formula
%                   - 'p-r':    Polak-Ribiere formula
%                   default = 'f-r'.
%   
%   - maxiter :     Maximum number of iterations. default = 500.
%
%   - tolgrad :     When inf-norm of the gradient is less than tolgrad,
%                   the optimization is considered as converged.
%                   default = 1.0e-6;
%
%   - tolx :        When the inf-norm of the change in x is less than 
%                   tolx, the optimization terminates.
%                   default = 1.0e-10;
%                                      
%   - tolfun :      When the change in objective value is less than 
%                   tolfun, the optimization terminates.
%                   default = 1.0e-10;
%
%   - display :     The level of displaying: 'off' | 'final' | {'iter'}.
%

%% main

% default options

opts.method = 'bfgs';
opts.cgscheme = 'f-r';
opts.maxiter = 500;
opts.tolgrad = 1.0e-6;
opts.tolx = 1.0e-10;
opts.tolfun = 1.0e-10;
opts.display = 'iter';

% override options

if ~isempty(varargin)
    
    if nargin == 1 && isstruct(varargin{1})
        new_opts = varargin{1};
        
        onames = fieldnames(new_opts);
        n = numel(onames);
        ovals = cell(1, n);
        for i = 1 : n
            ovals{i} = new_opts.(onames{i});
        end
        
    else
        onames = varargin(1:2:end);
        ovals = varargin(2:2:end);
        
        if ~(iscellstr(onames) && length(onames) == length(ovals))
            error('pli_optimset:invalidarg', ...
                'The input option list is invalid.');
        end
        
        n = numel(onames);
    end
    
    for i = 1 : n
        nam = lower(onames{i});
        v = ovals{i};
        
        switch nam
            case 'method'
                if ~(ischar(v))
                    error('pli_optimset:invalidarg', ...
                        'The value for method should be a string.');
                end
                
                v = lower(v);
                if ~any(strcmp(v, {'steepdesc', 'bfgs', 'cg', 'newton'}))
                    error('pli_optimset:invalidarg', ...
                        'The value for method is invalid.');
                end
                opts.method = v;
                
            case 'cgscheme'
                if ~(ischar(v))
                    error('pli_optimset:invalidarg', ...
                        'The value for cgscheme should be a string.');
                end
                
                v = lower(v);
                if ~any(strcmp(v, {'f-r', 'p-r'}))
                    error('pli_optimset:invalidarg', ...
                        'The value for cgscheme is invalid.');
                end
                opts.cgscheme = v;
                
            case 'maxiter'
                if ~(isnumeric(v) && isreal(v) && isscalar(v) && v >= 1)
                    error('pli_optimset:invalidarg', ...
                        'The value for maxiter should be a positive integer.');
                end
                opts.maxiter = v;
                
            case {'tolgrad', 'tolx', 'tolfun'}
                if ~(isfloat(v) && isreal(v) && isscalar(v) && v > 0)
                    error('pli_optimset:invalidarg', ...
                        'The value for %s should be a positive real value.', nam);
                end
                opts.(nam) = v;
                
            case 'display'
                if ~(ischar(v))
                    error('pli_optimset:invalidarg', ...
                        'The value for display should be a string.');
                end
                
                v = lower(v);
                if ~any(strcmp(v, {'off', 'final', 'iter'}))
                    error('pli_optimset:invalidarg', ...
                        'The value for display is invalid.');
                end
                opts.display = v;
                
            otherwise
                error('pli_optimset:invalidarg', ...
                    'Invalid option name %s', nam);
                
        end
        
    end
    
end


