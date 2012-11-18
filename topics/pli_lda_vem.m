function [R, objv] = pli_lda_vem(H, U, varargin)
%PLI_LDA_VEM Variational EM for Latent Dirichlet Allocation
%
%   R = PLI_LDA_VEM(H, U, ...);
%
%       Performs Variational EM to learn model parameters for 
%       Latent Dirichlet Allocation (LDA).
%
%   [R, objv] = PLI_LDA_VEM( ... );
%
%       Additionally returns the objective value.
%
%   Arguments
%   ---------
%   - H :       The input corpus, characterized by a histogram matrix.
%               H is a V-by-N matrix, each column corresponds to a
%               document.
%
%   - U :       The (initial guess of) word distributions of all topics. 
%               U is a K-by-V matrix.
%
%   Returns
%   -------
%   R is a result struct with the following fields:
%   - K :           The number of topics
%   - V :           The size of vocabulary
%   - N :           The number of documents
%
%   - alpha :       Dirichlet prior parameter
%   - U :           The word distributions of all topics [K-by-V]
%   - logU :        Logarithm of U
%
%   - Gam :         The gamma-vectors of all documents (K-by-N)
%   - TocW :        The topic weights of all documents (K-by-N)
%   - APhi :        The accumulated phi matrix (K-by-V)
%
%   
%   One can specify additional options as name/value pairs to control
%   the optimization procedure
%
%   Options
%   -------
%   - fixU :        Whether to fix U, the word distributions through
%                   iterations. (default = false)
%
%   - init_alpha :  Initial guess of alpha
%
%   - pricnt :      The prior count for each word (used to smooth
%                   the estimator of word distribution per topic)
%                   (default = 1)
%   
%   - maxiter :     The maximum number of iterations (default = 300)
%
%   - tolfun :      The tolerance of objective function changes upon
%                   convergence. (default = 1.0e-8 * N)
%
%   - inf_updates :  The number of inference updates per document in 
%                    each iteration (default = 10)
%
%   - display :     'off' | 'final' | {'iter'}
%


%% argument checking

if ~(isfloat(H) && isreal(H) && ismatrix(H))
    error('pli_lda_vem:invalidarg', 'H should be a real matrix.');
end
[V, N] = size(H);


if ~(isfloat(U) && isreal(U) && ismatrix(U))
    error('pli_lda_vem:invalidarg', 'U should be a real matrix.');
end
[K, Vu] = size(U);
if Vu ~= V
    error('pli_lda_vem:invalidarg', ...
        'The dimensions of U and H are inconsistent.');
end

s.fixU = false;
s.init_alpha = [];
s.pricnt = 1;
s.maxiter = 300;
s.tolfun = 1.0e-8 * N;
s.inf_updates = 10;
s.display = 'iter';

if ~isempty(varargin)
    s = pli_parseopts(s, varargin);
end


%% main

fixU = s.fixU;
pricnt = s.pricnt;
inf_updates = s.inf_updates;

% Initialize solution

if isempty(s.init_alpha)
    sol.alpha = ones(K, 1);
else
    sol.alpha = s.init_alpha;
end

sol.U = U;
sol.logU = log(U);
sol.Gam = [];

% optimization

[sol, objv] = pli_iteroptim('maximize', @vem_eval, @vem_update, sol, ...
    'maxiter', s.maxiter, ...
    'tolfun', s.tolfun, ...
    'nupdates', 1, ...
    'display', s.display);

% output

R.K = K;
R.V = V;
R.N = N;

R.alpha = sol.alpha;
R.U = sol.U;
R.logU = sol.logU;

R.Gam = sol.Gam;
R.TocW = sol.TocW;
R.APhi = sol.APhi;


%% core functions

    function s = vem_update(s)
        
        logU = s.logU;
        alpha = s.alpha;
        
        % prepare gamma
        
        Gam = s.Gam;
        if isempty(Gam)
            Gam = bsxfun(@plus, ones(K, 1) * (sum(H, 1) / K), alpha);
            Elog_theta = zeros(K, N);
        end
        
        % per-doc inference
        
        APhi = zeros(K, V);
        TocW = zeros(K, N);
        
        gam_ents = zeros(1, N);
        phi_ents = zeros(1, N);
        
        for i = 1 : N
            h = H(:, i);
            gam = Gam(:, i);
                        
            for j = 1 : inf_updates
            
                % update phi
                
                log_phi = bsxfun(@plus, logU, psi(gam));
                phi = pli_softmax(log_phi, 1);
                
                toc_w = phi * h;
                
                % update gamma
                
                gam = alpha + toc_w;
            end
            
            % store/accumulate results
            
            Gam(:, i) = gam;
            elog_t = psi(gam) - psi(sum(gam));
            Elog_theta(:, i) = elog_t;
            
            TocW(:, i) = toc_w;
            APhi = APhi + bsxfun(@times, h.', phi);
            
            gam_ents(i) = pli_dirichlet_entropy(gam);
            phi_ents(i) = pli_ddentropy(phi, 1) * h;
        end
        
        s.Gam = Gam;
        s.Elog_theta = Elog_theta;
        s.APhi = APhi;
        s.TocW = TocW;
        
        s.gamma_ents = gam_ents;
        s.phi_ents = phi_ents;
        
        % estimate alpha
        
        s.alpha = pli_dirichlet_mle(Elog_theta, alpha, ...
            'is_log', true, 'verbose', false);
        
        % estimate U
        
        if ~fixU
            if pricnt == 0
                A = APhi;
            else
                A = APhi + pricnt;
            end
            
            U = bsxfun(@times, A, 1 ./ sum(A, 2));
            
            s.U = U;
            s.logU = log(U);
        end
        
    end

    
    function objv = vem_eval(s)

        Elog_theta = s.Elog_theta;
        logU = s.logU;
        
        % E log(theta | alpha)
        
        lnB = pli_mvbetaln(s.alpha);
        ell_theta = (s.alpha - 1)' * Elog_theta - lnB;
        
        % E log(z | theta)
        
        ell_z = sum(s.TocW .* Elog_theta, 1);
        
        % E log(w | z; U)
        
        sum_ell_w = sum(sum(s.APhi .* logU, 1));
        
        % log(U | pri_c)
        
        if pricnt > 0
            lpri_U = sum(sum(logU, 1)) * pricnt;
        else
            lpri_U = 0;
        end
        
        % overall
        
        objv = sum(ell_theta) + ...
            sum(ell_z) + ...
            sum_ell_w + ...
            sum(s.gamma_ents) + ...
            sum(s.phi_ents) + ...
            lpri_U;
                
    end


end

