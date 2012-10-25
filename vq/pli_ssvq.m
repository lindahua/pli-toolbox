function [C, w, cbnd] = pli_ssvq(X, sw, cbnd, kmax, varargin)
%PLI_SSVQ Stochastic Streaming vector quantization
%
%   [C, w] = PLI_SSVQ(X, sw, cbnd, kmax, ...);
%
%       Performs stochastic streaming vector quantization as follows.
%
%       Starting from the current set of centers given by C and their
%       associated weights given by w, the function runs the online
%       facility location algorithm until either all samples in X are 
%       visited or the number of centers reach kmax.
%
%       If the number of centers reaches kmax, and there remain unvisited
%       samples, it consolidates the current set of centers, which an 
%       gradually increased cost bound (each iteration cbnd is increased 
%       to cbnd * (1 + beta)), until the number of centers are brought 
%       down to kmax / 2.
%
%       This process continues until are samples are processed.
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
%
%   In addition, one may specify options in the form of name/value pairs
%   to control the process.
%
%   Options
%   -------
%   - init :    The init set of weighted centers, in the form as 
%               {C, w}. Here, C is the matrix of centers, and w is 
%               the vector of associated weights.
%
%   - beta :    The ratio of cost bound increasing at each iteration of
%               the consolidation process. (default = 0.2).
%
%   - shrink :  The shrinking ratio for consolidation. (default = 0.75).
%
%   - weights : The weights of samples, which can be either empty,
%               indicating that all samples have unit weights, or a vector
%               of length n.
%               default is [].
%
%   - vb_intv : Display progress after every vb_intv samples have been
%               processed. To suppress all display, set this to 0.
%               Default = max(1, min(1000, n / 200)), here n = size(X, 2).
%   
%   Remarks
%   -------
%     - It is vitally important to first randomly permute the data set
%       before running SSVQ.
%
%     - One can handle a huge data set that cannot be hosted entirely
%       in memory by dividing it into several batches, and perform the
%       vector quantization by repeatedly invoking this function to 
%       update the centers, as follows
%
%           C = []; w = [];
%           for i = 1 : num_batches
%               X = ... load the i-th batch ...
%               [C, w, cbnd] = PLI_SSVQ(C, w, cbnd, X, kmax);
%           end
%
%       Another strategy will be to first respectively run SSVQ on each
%       patch and then combine them (and run SSVQ on a combined set if
%       necessary).
%


%% argument checking

if ~(isfloat(X) && isreal(X) && ismatrix(X) && ~issparse(X))
    error('pli_ssvq:invalidarg', ...
        'X should be a non-sparse real matrix.');
end
[d, n] = size(X);


if isempty(sw)
    sw = ones(1, n);
else
    if ~(isfloat(sw) && isreal(sw) && ~issparse(sw) && isvector(sw))
        error('pli_ssvq:invalidarg', ...
            'weights should be a real vector.');
    end
    
    if length(sw) ~= n
        error('pli_ssvq:invalidarg', 'weights should be of length n.');
    end            
end

if ~(isfloat(cbnd) && isreal(cbnd) && cbnd > 0)
    error('pli_ssvq:invalidarg', ...
        'cbnd should be a positive real value.');
end

if ~(isnumeric(kmax) && isreal(kmax) && kmax == fix(kmax) && kmax > 1)
    error('pli_ssvq:invalidarg', ...
        'kmax should be a positive integer with kmax > 1.');
end

% parse options

opts.init = {[], []};
opts.beta = 0.2;
opts.shrink = 0.75;
opts.weights = [];
opts.vb_intv = max(100, min(10000, n / 200));

if ~isempty(varargin)
    opts = pli_parseopts(opts, varargin);
end

vb_intv = opts.vb_intv;


% check C and w

if ~(iscell(opts.init) && numel(opts.init) == 2)
    error('pli_ssvq:invalidarg', ...
        'The option init should be a cell arry in the form of {C, w}.');
end

C = opts.init{1};
w = opts.init{2};

if ~(isfloat(C) && isreal(C) && ismatrix(C) && ~issparse(C))
    error('pli_ssvq:invalidarg', ...
        'Init C should be a non-sparse real matrix or empty.');
end

if isempty(C)
    if ~isempty(w)
        error('pli_ssvq:invalidarg', ...
            'Init w should be empty when C is empty.');
    end
else
    if ~(isfloat(w) && isvector(w) && ~issparse(w) && numel(w) == size(C,2))
        error('pli_ssvq:invalidarg', ...
            'Init w should be a real vector of length size(C,2).');
    end
end

if ~isempty(C)
    if size(C, 1) ~= d
        error('pli_ssvq:invalidarg', ...
            'Sample dimensions in C and X are inconsistent.');
    end
    
    if kmax < size(C, 2)
        error('pli_ssvq:invalidarg', 'size(C, 2) exceeds kmax.');
    end
end


%% main

% pre-process inputs

if ~isa(X, 'double'); X = double(X); end
if ~isa(sw, 'double'); sw = double(sw); end

if ~isempty(C)
    if ~isa(C, 'double'); C = double(C); end
    if ~isa(w, 'double'); w = double(w); end
end


% main loop

i = 0;  % the number of samples that have been processed

if vb_intv
    fprintf('Stream-VQ (n = %d, kmax = %d): \n', n, kmax);
end

u = rand(1, n);

while i < n
    
    if i > 0
        % consolidate existing centers
        
        kgoal = max(1, min(kmax - 1, round(kmax * opts.shrink)));
                
        if vb_intv
            fprintf('Consolidation (goal %d ==> %d): \n', size(C,2), kgoal);
        end
                        
        iround = 0;

        while size(C, 2) > kgoal
            iround = iround + 1;
            cbnd = cbnd * (1 + opts.beta);
            
            uc = rand(1, size(C, 2));
            
            [C, w, ~] = ...
                ssvq_cimp([], [], cbnd, C, w, uc, size(C,2), 0, vb_intv);
            
            if vb_intv
                fprintf('    round %d: K = %d, cbnd = %.4g\n', ...
                    iround, size(C, 2), cbnd);
            end
        end                
    end
    
    % scan remaining samples
    
    if vb_intv
        fprintf('Stream processing:\n');
    end
    
    [C, w, i] = ssvq_cimp(C, w, cbnd, X, sw, u, kmax, i, vb_intv); 
            
end

if vb_intv
    fprintf('Stream-VQ completed: K = %d\n', size(C, 2));
end


