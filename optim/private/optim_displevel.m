function v = optim_displevel(sz)
%OPTIM_DISPLEVEL Get display level integer from string
%
%   v = OPTIM_DISPLEVEL(sz);
%

switch sz
    case 'off'
        v = 0;
    case 'final'
        v = 1;
    case 'iter'
        v = 2;
    otherwise
        error('pli_optimset:invalidarg', ...
            'The value for display is invalid.');
end
