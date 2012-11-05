function lst = pli_mex_targets()
%PLI_MEX_TARGETS The List of mex targets for PLI toolbox
%
%   lst = PLI_MEX_TARGETS
%
%       Returns the list of all mex targets in PLI toolbox
%
%       lst is an array of structs, each corresponding to a target.
%       The struct contains following fields:
%
%           - name:     The name of the target
%           - path:     The relative output path
%           - sources:  a string or a cell array of strings
%

%% main

% use $ as a short-hand of the output path

lst_common = { ...
    'repnum_cimp',    'common/private', '$/repnum_cimp.cpp'; ...
    'intcount1_cimp', 'common/private', '$/intcount1_cimp.cpp'; ...
    'intcount2_cimp', 'common/private', '$/intcount2_cimp.cpp'; ...
    'aggregx_cimp',   'common/private', '$/aggregx_cimp.cpp'; ...
    'aggreg2_cimp',   'common/private', '$/aggreg2_cimp.cpp'; ...
    'findbin_cimp',   'common/private', '$/findbin_cimp.cpp'; ...
    'idxgroup_cimp',  'common/private', '$/idxgroup_cimp.cpp'; ...
    'ktop_cimp',      'common/private', '$/ktop_cimp.cpp' ...
};    

lst_metrics = { ...
    'pw_metrics_cimp',   'metrics/private', '$/pw_metrics_cimp.cpp'; ...
    'pw_minkowski_cimp', 'metrics/private', '$/pw_minkowski_cimp.cpp' ...
};

lst_optim = { ...
    'lbfgs_calcdir_cimp',  'optim/private', '$/lbfgs_calcdir_cimp.cpp' ...
};

lst_vq = { ...
    'kmpp_cimp',  'vq/private', '$/kmpp_cimp.cpp'; ...
    'ssvq_cimp',  'vq/private', '$/ssvq_cimp.cpp'
};

lst_svm = { ...
    'pegasos_cimp',  'svm/private', '$/pegasos_cimp.cpp'
    'sgdqn_cimp',    'svm/private', '$/sgdqn_cimp.cpp'
};

lst = [lst_common; lst_metrics; lst_optim; lst_vq; lst_svm];

lst = cell2struct(lst, ...
    {'name', 'path', 'sources'}, 2);

for i = 1 : numel(lst)
    lst(i) = post_process(lst(i));
end

%% auxiliary function

function t = post_process(t)

if ischar(t.sources)
    t.sources = strrep(t.sources, '$', t.path);
else
    for i = 1 : numel(t.sources)
        t.sources{i} = strrep(t.sources{i}, '$', t.path);
    end
end
