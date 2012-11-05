function plidemo_kersvm(n, solver)
%PLIDEMO_KERSVM Demos the use of pli_kersvm
%
%   PLIDEMO_KERSVM(n, solver);
%
%       Here, n is the number of points in each class, and solver
%       is either 'ip' or 'gurobi'
%

%% arguments

if nargin < 1
    n = 1000;
end

if nargin < 2
    solver = 'ip';
end

%% data generation

sigma = 0.1;
Ksig = 1;

Xp = rand_circle(n, 1, sigma);
Xn = rand_circle(n, 2, sigma);

X = [Xp, Xn];
y = [ones(1, n), -ones(1, n)];

%% solve SVM

K = pli_kernel('gauss', X, [], Ksig);
C = 1;

opts = [];
if ischar(solver)
    switch solver
        case 'ip'
            opts = optimset('Display', 'iter');
        case 'gurobi'
            opts.outputflag = 1;
    end
end

[alpha, bias] = pli_kersvm(K, y, C, solver, opts);

%% visualization

figure;
plot(Xp(1,:), Xp(2,:), 'r.');
hold on;
plot(Xn(1,:), Xn(2,:), 'b.');
axis equal;

xx = -2.5 : 0.1 : 2.5;
yy = -2.5 : 0.1 : 2.5;
[xx, yy] = meshgrid(xx, yy);

gX = [xx(:)'; yy(:)'];
gK = pli_kernel('gauss', X, gX, Ksig);

u = pli_kersvm_predict(alpha, bias, y, gK);
u = reshape(u, size(xx));

[c, h] = contour(xx, yy, u, [-1, 0, 1]);
clabel(c, h);


%% main

function X = rand_circle(n, r, sigma)

t = rand(1, n) * (2 * pi);
c = cos(t);
s = sin(t);

r = r + randn(1, n) * sigma;
x = r .* c;
y = r .* s;

X = [x; y];


