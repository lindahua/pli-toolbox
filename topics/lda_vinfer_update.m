function [gam, phi, toc_w, elog_t] = lda_vinfer_update(alpha, logU, h, gam, T)
%LDA_VINFER_UPDATE Performs LDA Variational inference update
%
%   [gam, phi, toc_w, elog_t] = LDA_VINFER_UPDATE(alpha, logU, h, gam, niters)
%
%       Updates the variational latent variable: gamma and phi, for
%       a document.
%
%   Arguments
%   ---------
%   - alpha :       The Dirichlet prior, size = [K, 1]
%   - logU :        The log of unigrams, size = [V, K]
%   - h :           The document histogram, size = [V, 1]
%   - gam :         The gamma variable, size = [K, 1]
%   - T :           The number of iterations
%
%   Returns
%   -------
%   - gam :         Updated gamma variable, size = [K, 1]
%   - phi :         Updated phi variable, size = [V, K]
%   - toc_w :       Topic weights (i.e. phi' * h), size = [K, 1]
%   - elog_t :      E[log(theta)] w.r.t gamma, size = [K, 1]
%   
%   Notes
%   -----
%       This function is supposed to be called within an iterative
%       optimization procedure. For the sake of efficiency, 
%       no argument checking is performed.
%

%% main

for i = 1 : T
    % update phi
    
    log_phi = bsxfun(@plus, logU, psi(gam).');
    phi = normalize_exp(log_phi, 2);
    
    toc_w = (h' * phi).';
    
    % update gamma
    
    gam = alpha + toc_w;
end

if nargout >= 4
    elog_t = psi(gam) - psi(sum(gam));
end

