function pli_build(varargin)
%PLI_BUILD Build C-mex functions of the PLI toolbox
%
%   PLI_BUILD
%       builds all targets
%
%   PLI_BUILD(target)
%       builds a specific target
%
%   PLI_BUILD(target1, target2, ...)
%       builds a collection of targets
%
%   PLI_BUILD('-debug', target)
%       builds a specific target using debug options 
%       (e.g. turning on index checks)
%       

%% Examine environment

lmat_home = getenv('LIGHT_MATRIX_HOME');
if isempty(lmat_home)
    error('pli_build:rterror', ...
        'Environment variable LIGHT_MATRIX_HOME is not set.');
end


%% Extract targets

all_targets = pli_mex_targets();

if isempty(varargin)    
    targets = all_targets;
    use_debug = 0;
    
else
    assert(iscellstr(varargin), ...
        'pli_build:invalidarg', 'all arguments must be strings.');
    
    if strcmpi(varargin{1}, '-debug')
        assert(numel(varargin) == 2, ...
            'pli_build:invalidarg', ...
            'Debug-mode should apply to exactly one target');
        
        targets = find_target(all_targets, varargin{2});        
        use_debug = 1;
    else
        n = numel(varargin);
        targets = cell(n, 1);
        for i = 1 : n
            targets{i} = find_target(all_targets, varargin{i});
        end
        targets = vertcat(targets{:});
        use_debug = 0;
    end
    
end

%% Perform compilation

pli_home = fileparts(mfilename('fullpath'));

for i = 1 : numel(targets)
    build_target(targets(i), pli_home, lmat_home, use_debug);
end

disp(' ');

%% Building

function build_target(t, pli_home, lmat_home, use_debug)

fprintf('Building %s ...\n', t.name);

% compilation options

if use_debug
    opts = {'-DLMAT_DIAGNOSIS_LEVEL=3'};
else
    opts = {'-O', '-DLMAT_DIAGNOSIS_LEVEL=0'};
end

out_path = fullfile(pli_home, t.path);
opts = [opts, {['-I' lmat_home], '-outdir', out_path}];

% Go!

if ischar(t.sources)
    mex(opts{:}, t.sources);
else
    mex(opts{:}, t.sources{:});
end


%% Target extraction

function t = find_target(all_targets, tname)

n = numel(all_targets);

t = [];
for i = 1 : n
    ct = all_targets(i);
    if strcmp(tname, ct.name)
        t = ct;
    end
end

if isempty(t)
    error('pli_build:invalidarg', ...
        'Target %s not found.', tname);
end


