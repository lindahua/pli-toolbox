function [gam, phi, objv] = pli_lda_vinfer(alpha, logU, h, varargin)
%PLI_LDA_VINFER Variational Inference for Latent Dirichlet Allocation
%
%   [gam, phi] = PLI_LDA_VINFER(alpha, logU, h, ...);
%
%       Performs variational inference based on Latent Dirichlet 
%       Allocation for a document.
%
%   [gam, phi, objv] = PLI_LDA_VINFER( ... );
%
%       Additionally returns the objective value at last step.
%
%
%   Arguments
%   ---------
%   Suppose there are K topics over a vocabulary of size V.
%
%   - alpha :       The Dirichlet prior parameter
%                   (which can be either a scalar or a vector of length K)
%
%   - logU :        The logarithm of word probabities of each topic.
%                   It should be a K-by-V matrix.
%
%   - h :           The word histogram
%                   It should be a vector of length V, which can be 
%                   full or sparse.
%
%   Returns
%   -------
%   - gam :         The inferred posterior Dirichlet parameter for
%                   topic distribution. Size: K-by-1.
%
%   - phi :         The per-word posterior distribution of topics. 
%                   Size: K-by-V.
%
%
%   One can specify options to control the optimization procedure.
%
%   Options
%   -------
%   - init_gam :    The initial guess of gamma. 
%   
%   - maxiter :     The maximum number of iterations (default = 100)
%
%   - tolfun :      The tolerance of objective function changes upon
%                   convergence. (default = 1.0e-10)
%
%   - nupdates :    The number of updates in each iteration (default = 1)
%
%   - display :     {'off'} | 'final' | 'iter'
%

%% argument checking

if ~(isfloat(alpha) && isreal(alpha) && isvector(alpha))
    error('pli_lda_vinfer:invalidarg', ...
        'alpha should be either a scalar or a real vector.');
end

if size(alpha, 2) > 1
    alpha = alpha.';
end

if ~(isfloat(logU) && ismatrix(logU))
    error('pli_lda_vinfer:invalidarg', 'logU should be a real matrix.');
end

[K, V] = size(logU);

if ~(isscalar(alpha) || size(alpha, 1) == K)
    error('pli_lda_vinfer:invalidarg', 'The size of alpha is invalid.');
end

if ~(isfloat(h) && isreal(h) && isequal(size(h), [V 1]))
    error('pli_lda_vinfer:invalidarg', 'h should be a V-by-1 real vector.');
end

s.init_gam = [];
s.maxiter = 100;
s.tolfun = 1.0e-10;
s.nupdates = 1;
s.display = 'off';

if ~isempty(varargin)
    s = pli_parseopts(s, varargin);
end

%% main

% initialize

if isempty(s.init_gam)            
    sol.gam = alpha + sum(h) / K;
    if isscalar(sol.gam)
        sol.gam = sol.gam * ones(K, 1);
    end
else
    sol.gam = s.init_gam;
end

[sol, objv] = pli_iteroptim('maximize', @vi_eval, @vi_update, sol, ...
    'maxiter', s.maxiter, ...
    'tolfun', s.tolfun, ...
    'nupdates', s.nupdates, ...
    'display', s.display);

gam = sol.gam;
phi = sol.phi;


%% core functions

    function s = vi_update(s0)
        
        % update phi
        
        log_phi = bsxfun(@plus, logU, psi(s0.gam)); % K-by-V
        s.phi = pli_softmax(log_phi, 1);              % K-by-V
        
        s.toc_w = s.phi * h;    % K-by-1
        
        % update gamma
        
        s.gam = alpha + s.toc_w;      
    end


    function objv = vi_eval(s)        
        
        elog_t = psi(s.gam) - psi(sum(s.gam));
        
        % E log(theta | alpha)
        
        if isscalar(alpha)
            if alpha == 1
                ell_theta = 0;
            else
                ell_theta = (alpha - 1) * sum(elog_t);
            end
        else
            ell_theta = (alpha - 1)' * elog_t;
        end
        
        % E log(z | theta)
        
        ell_z = s.toc_w' * elog_t;
        
        % E log(w | z; U)
        
        ell_w = sum((s.phi .* logU) * h);
        
        % entropy of gamma
        
        ent_gam = pli_dirichlet_entropy(s.gam, elog_t);
        
        % entropy of phi
        
        ent_phi = pli_ddentropy(s.phi, 1) * h;
        
        % overall
        
        objv = ell_theta + ell_z + ell_w + ent_gam + ent_phi;
    end


end

