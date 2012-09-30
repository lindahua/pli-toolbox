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
    'intcount1_cimp', 'common/private', '$/intcount1_cimp.cpp'; ...
    'intcount2_cimp', 'common/private', '$/intcount2_cimp.cpp'; ...
    'aggregx_cimp', 'common/private', '$/aggregx_cimp.cpp'; ...
    'aggreg2_cimp', 'common/private', '$/aggreg2_cimp.cpp' ...
};    


lst = lst_common;

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
