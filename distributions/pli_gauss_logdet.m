function v = pli_gauss_logdet(G)
%PLI_GAUSS_LOGDET Log-determinant of Gaussian covariance
%
%   v = PLI_GAUSS_LOGDET(G);
%
%       Evaluates the log-determinant of the covariance(s) in G.
%
%   Arguments
%   ---------
%   - G :   A Gaussian distribution struct.
%   
%   Returns
%   -------
%   - v:    The result(s). If there is a single distribution in G or
%           the covariance is tied, then v is a scalar, otherwise 
%           the size of v is [m, 1], where m == G.num.
%

%% main

cov = G.cov;

switch G.cform
    case 0
        v = log(cov) * G.dim;
    case 1
        v = sum(log(cov), 1);
        if size(cov, 2) > 1
            v = v.';
        end
    case 2
        if ismatrix(cov)
            v = pli_logdet(cov);
        else
            m = G.num;
            v = zeros(m, 1);
            for k = 1 : m
                v(k) = pli_logdet(cov(:,:,k));
            end
        end
end
        

