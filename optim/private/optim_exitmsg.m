function msg = optim_exitmsg(exitflag)
% Get the message of exiting condition according to exitflag
%
%   msg = optim_exitmsg(exitflag)
%

switch exitflag
    case 0
        msg = 'Maximum iterations reached';
    case 1
        msg = 'Converged (first-order condition satisfied)';
    case 2
        msg = 'Change in solution was too small';
    case 3
        msg = 'Change in objective value was too small';
    case 5
        msg = 'Failed to decrease the objective along search direction';
    case -3
        msg = 'The objective value is not bounded.';
end

