function plidemo_linsvm(n, solver)
%PLIDEMO_LINSVM Demonstrates the use of linear SVM
%
%   PLIDEMO_LINSVM();
%   PLIDEMO_LINSVM(n);
%   PLIDEMO_LINSVM(n, solver);
%
%       Here, n is the number of samples of each class.
%       The default value of n is 500;
%
%       solver is either 'ip', 'gurobi'.
%

%% arguments

if nargin < 1
    n = 500;
end

if nargin < 2
    solver = 'ip';
end

%% Data generation

t = deg2rad(60);
R = [cos(t) -sin(t); sin(t) cos(t)];
Z = R * diag([4, 1]) * randn(2, 2 * n);

Xp = bsxfun(@plus, Z(:, 1:n), [5 0]');
Xn = bsxfun(@plus, Z(:, n+1:2*n), [-5 0]');

X = [Xp, Xn];
y = [ones(1, n), -ones(1, n)];

%% Solve SVM

c = 10;

% set to show optimization procedure

opts = [];

if strcmp(solver, 'ip')
    opts = optimset('Display', 'iter');
elseif strcmp(solver, 'gurobi')
    opts.outputflag = 1;
end

% solve

tic;
[w, w0, ~, objv] = pli_linsvm(X, y, c, solver, opts);
solve_time = toc;

fprintf('Solve time = %.4f sec\n', solve_time); 

% show solution

disp('Solutions');
disp('=============');
display(w);
display(w0);
display(objv);

%DEBUG
% objv0 = 0.5 * sum(w.^2) + c * sum(max(0, 1 - y .* (w' * X + w0)));
% fprintf('objv.dev = %g\n', objv0 - objv);


%% Visualization

figure;
plot(Xp(1,:), Xp(2,:), 'g.');
hold on;
plot(Xn(1,:), Xn(2,:), 'm.');

axis equal;

px = X(1,:);
py = X(2,:);
rgn = [min(px), max(px), min(py), max(py)];

drawline(rgn, w, w0, 'Color', 'b');
drawline(rgn, w, w0 - 1.0, 'Color', [0 0.5 0]);
drawline(rgn, w, w0 + 1.0, 'Color', [0.5 0 0]);

r = y .* (w' * X + w0);
sv = find(r < 1 + 1.0e-6);
if ~isempty(sv)
    hold on;
    plot(X(1, sv), X(2, sv), 'ro', 'MarkerSize', 10);
end


function drawline(rgn, w, w0, varargin)
% Draws a line on current axis
%
%   rgn = [xmin, xmax, ymin, ymax];
%
%   Draw line: w(1) * x + w(2) * y + w0 = 0;

xmin = rgn(1);
xmax = rgn(2);
ymin = rgn(3);
ymax = rgn(4);

w1 = w(1);
w2 = w(2);

if abs(w1) < abs(w2)
    
    x0 = xmin - 0.1 * (xmax - xmin);
    x1 = xmax + 0.1 * (xmax - xmin);
    
    y0 = -(w1 * x0 + w0) / w2;
    y1 = -(w1 * x1 + w0) / w2;
    
else
    y0 = ymin - 0.1 * (ymax - ymin);
    y1 = ymax + 0.1 * (ymax - ymin);
    
    x0 = -(w2 * y0 + w0) / w1;
    x1 = -(w2 * y1 + w0) / w1;
    
end

line([x0, x1], [y0, y1], varargin{:});



