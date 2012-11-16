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

cvals = G.cvals;

switch G.cform
    case 's'
        v = log(cvals) * G.dim;
    case 'd'
        v = sum(log(cvals), 1);
        if size(cvals, 2) > 1
            v = v.';
        end
    case 'f'
        if ismatrix(cvals)
            v = pli_logdet(cvals);
        else
            m = G.num;
            v = zeros(m, 1);
            for k = 1 : m
                v(k) = pli_logdet(cvals(:,:,k));
            end
        end
end
        

