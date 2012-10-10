function plidemo_logireg(n)
%PLIDEMO_LOGIREG Demonstrates the use of logistic regression
%
%   PLIDEMO_LOGIREG;
%   PLIDEMO_LOGIREG(n);
%
%       Here, n is the number of samples. (default n = 500);
%

%% arguments

if nargin < 1
    n = 500;
end

%% Data generation

t = pi / 3;
R = [cos(t) -sin(t); sin(t) cos(t)];
Z = R * diag([4, 1]) * randn(2, 2 * n);

xcen = 5;

Xp = bsxfun(@plus, Z(:, 1:n), [-5 + xcen, 0]');
Xn = bsxfun(@plus, Z(:, n+1:2*n), [5 + xcen, 0]');

X = [Xp, Xn];
y = [ones(1, n), -ones(1, n)];

%% Solve

lambda = 0.01 * n;
lambda0 = lambda * 0.02;

opts = pli_optimset('gd', 'display', 'iter');
solver = @(f, x) pli_fmingd(f, x, opts);

tic;
[theta, bias] = pli_logireg(X, y, [], lambda, lambda0, [], solver);
solve_time = toc;

fprintf('Solve time = %f sec\n', solve_time);

%% Visualization

figure;
plot(Xp(1,:), Xp(2,:), 'g.');
hold on;
plot(Xn(1,:), Xn(2,:), 'm.');

axis equal;

px = X(1,:);
py = X(2,:);
rgn = [min(px), max(px), min(py), max(py)];

drawline(rgn, theta, bias, 'Color', 'b');
drawline(rgn, theta, bias - 3.0, 'Color', [0 0.5 0]);
drawline(rgn, theta, bias + 3.0, 'Color', [0.5 0 0]);


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

