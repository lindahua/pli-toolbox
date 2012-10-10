function opt = pli_optimset(opt0, varargin)
%PLI_OPTIMSET Sets the optimization options
%
%   opt = PLI_OPTIMSET(fun);
%
%       Gets the default options for a function. Here, fun is a string,
%       which can take following values
%
%       - 'gd':         for pli_fmingd
%       - 'bfgs':       for pli_fminbfgs
%       - 'newton':     for pli_fminnewton
%
%   opt = PLI_OPTIMSET(fun, 'name1', val1, 'name2', val2, ...);
%   
%       Overrides the options for a function.
%
%   opt = PLI_OPTIMSET(opt, 'name1', val1, 'name2', vals, ...);
%
%       Overrides a given option struct.
%
%
%   List of supported options:
%   
%   - maxiter :     Maximum number of iterations
%   - tolx :        Tolerable changes in x at convergence
%   - tolfun :      Tolerable changes in objective values at convergence
%   - display :     The level of displaying: 'off' | 'final' | 'iter'
%   - backtrk :     The back tracking coefficient in line search.
%

%% main

% default options

if ischar(opt0)
    fun = opt0;
        
    switch fun
        case 'gd'
            opt.maxiter = 500;
            opt.tolx = 1.0e-6;
            opt.tolfun = 1.0e-7;
            opt.backtrk = 0.5;
            opt.display = 'off';
            
        case 'bfgs'
            opt.maxiter = 150;
            opt.tolx = 1.0e-7;
            opt.tolfun = 1.0e-8;
            opt.backtrk = 0.5;
            opt.display = 'off';     
            
        case 'newton'
            opt.maxiter = 100;
            opt.tolx = 1.0e-8;
            opt.tolfun = 1.0e-9;
            opt.backtrk = 0.75;
            opt.display = 'off';            
            
        otherwise
            error('pli_optimset:invalidarg', ...
                'Unsupported function name %s', fun);
    end
    
elseif isstruct(opt0) && numel(opt0) == 1
    
    opt = opt0; 
    
end

% override options

if ~isempty(varargin)
    
    onames = varargin(1:2:end);
    ovals = varargin(2:2:end);
    
    if ~(iscellstr(onames) && length(onames) == length(ovals))
        error('pli_optimset:invalidarg', ...
            'The option list is invalid.');
    end
    
    n = numel(onames);
    for i = 1 : n
        nam = lower(onames{i});
        v = ovals{i};
        
        switch nam
            case 'maxiter'
                if ~(isnumeric(v) && isreal(v) && isscalar(v) && v >= 1)
                    error('pli_optimset:invalidarg', ...
                        'The value for maxiter should be a positive integer.');
                end
                opt.maxiter = v;
                
            case 'tolx'
                if ~(isfloat(v) && isreal(v) && isscalar(v) && v > 0)
                    error('pli_optimset:invalidarg', ...
                        'The value for tolx should be a positive real value.');
                end                
                opt.tolx = v;
                
            case 'tolfun'
                if ~(isfloat(v) && isreal(v) && isscalar(v) && v > 0)
                    error('pli_optimset:invalidarg', ...
                        'The value for tolfun should be a positive real value.');
                end                
                opt.tolfun = v; 
                
            case 'display'
                if ~(ischar(v))
                    error('pli_optimset:invalidarg', ...
                        'The value for display should be a string.');
                end                
                opt.display = v;                
                                            
            case 'backtrk'
                if ~(isfloat(v) && isreal(v) && isscalar(v) && v > 0 && v < 1)
                    error('pli_optimset:invalidarg', ...
                        'The value for backtrk should be a real value in (0, 1).');
                end                
                opt.tolx = v;
        end
        
    end
    
end


