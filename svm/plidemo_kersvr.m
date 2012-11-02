function plidemo_kersvr(n, solver)
%PLIDEMO_KERSVR Demos the use of kernel SVR on a synthetic data set
%
%   PLIDEMO_KERSVR;
%   PLIDEMO_KERSVR(n);
%   PLIDEMO_KERSVR(n, solver);
%
%       Here, n is the number of samples (default = 1000), and
%       solver can be either 'ip', 'gurobi', or a user-supplied solver.
%

%% arguments

if nargin < 1
    n = 1000;
end

if nargin < 2
    solver = 'ip';
end

%% data generation

x = rand(1, n) * (4 * pi);
noise_sig = 0.2;
y = 2.0 + sin(x) + randn(size(x)) * noise_sig;

% compute Gaussian kernel

sigma = 0.5;
K = calc_ker(x, x, sigma);

% solve SVM

C = 10;
e = noise_sig;

opts = [];
if ischar(solver)
    switch solver
        case 'ip'
            opts = optimset('Display', 'iter');
        case 'gurobi'
            opts.outputflag = 1;
    end
end

tic;
[alpha, b] = pli_kersvr(K, y, e, C, solver, opts);
solve_time = toc;

fprintf('Solve time = %.4f sec\n', solve_time);


% make prediction

xt = linspace(0, 4*pi, 2000);
Kt = calc_ker(x, xt, sigma);
yt = alpha' * Kt + b;


%% visualization

figure;
plot(x, y, 'b.');
hold on;
plot(xt, yt, 'r-', 'LineWidth', 2);

axis([0, 4 * pi, 0, 4]);

%% Auxiliary function

function K = calc_ker(x, y, sigma)

x = x(:);
y = y(:).';

D = bsxfun(@minus, x, y);
K = exp(- (D.^2) * (1/(sigma^2)));
