function [C, w, cbnd] = pli_ssvq_mc(X, sw, cbnd, kmax, varargin)
%PLI_SSVQ_MC Stochastic Streaming Vector Quantization via Multi-Consolidation
%
%   [C, w, cbnd] = PLI_SSVQ_MC(X, sw, cbnd, kmax, ...);
%
%       Performs stochastic streaming vector quantization through
%       multiple consolidation.
%
%       At each stage, it applies SSVQ on a set of (weighted) samples
%       without constraint on the number of centers. If the number of
%       resultant centers is greater than kmax, the centers will be
%       used as input at next stage, where SSVQ will be applied with
%       an increased cost-bound. This procedure continues until all
%       the number of centers are reduced to below kmax.
%
%   Arguments
%   ----------
%   - X :       The samples to be processed: a matrix of size [d, n].
%
%   - sw :      The sample weights, which can be either empty or a 
%               vector of length n.
%
%   - cbnd :    The initial cost bound.
%
%   - kmax :    The maximum number of centers.
%               If one wants to remove the constraint on K, simply
%               set kmax to any number greater than the number of 
%               samples. 
%
%   Returns
%   -------
%   - C :       The set of centers.
%
%   - w :       The weights associated with the centers.
%
%   - cbnd :    The updated cost bound.
%
%   - kbound :  The hard bound on the number of K. (default = 5000);
%
%   In addition, one may specify options in the form of name/value pairs
%   to control the process.
%
%   Options
%   -------
%
%   - beta :    The ratio of cost bound increasing at each iteration 
%               (default = 0.2).
%
%   - vb_intv : Display progress after every vb_intv samples have been
%               processed. To suppress all display, set this to 0.
%               Default = max(1, min(1000, n / 200)), here n = size(X, 2).
%

%% argument checking

if ~(isfloat(X) && isreal(X) && ismatrix(X) && ~issparse(X))
    error('pli_ssvq_mc:invalidarg', ...
        'X should be a non-sparse real matrix.');
end
n = size(X, 2);

if isempty(sw)
    sw = ones(1, n);
else
    if ~(isfloat(sw) && isreal(sw) && ~issparse(sw) && isvector(sw))
        error('pli_ssvq_mc:invalidarg', ...
            'weights should be a real vector.');
    end
    
    if length(sw) ~= n
        error('pli_ssvq_mc:invalidarg', 'weights should be of length n.');
    end            
end

if ~(isfloat(cbnd) && isreal(cbnd) && cbnd > 0)
    error('pli_ssvq_mc:invalidarg', ...
        'cbnd should be a positive real value.');
end

if ~(isnumeric(kmax) && isreal(kmax) && kmax == fix(kmax) && kmax > 1)
    error('pli_ssvq_mc:invalidarg', ...
        'kmax should be a positive integer with kmax > 1.');
end

% parse options

opts.beta = 0.2;
opts.kbound = 5000;
opts.vb_intv = max(100, min(10000, n / 200));

if ~isempty(varargin)
    opts = pli_parseopts(opts, varargin);
end

kbound = opts.kbound;
beta = opts.beta;
vb_intv = opts.vb_intv;


%% main

% pre-process inputs

if ~isa(X, 'double'); X = double(X); end
if ~isa(sw, 'double'); sw = double(sw); end

% main loop

C = X;
w = sw;
cur_k = size(C, 2);

t = 0;

while cur_k > kmax
    
    t = t + 1;            
    if t > 1
        cbnd = cbnd * (1 + opts.beta);
    end
    
    if vb_intv
        fprintf('\nIteration %d (cbnd = %.4g):\n', t, cbnd);
    end
 
    u = rand(1, cur_k);
    kb = min(kbound, size(C, 2));
    [C, w] = pli_ssvq(C, w, cbnd, kb, 'vb_intv', vb_intv, 'beta', beta);
    cur_k = size(C, 2);
    
    if vb_intv
        fprintf('    K = %d\n', cur_k);
    end
end


