function demo_kmedoid(K, n)
%DEMO_KMEDOID A simple program to demonstrate the use of kmedoid
%
%   DEMO_KMEDOID;
%   DEMO_KMEDOID(K);
%   DEMO_KMEDOID(K, n);
%
%       Perform K-medoid on n randomly generated points.
%
%   Arguments:
%   ----------
%   - K :       The number of centers (default = 20)
%   - n :       The total number of points (default = K * 100).
%

%% arguments

if nargin < 1
    K = 3;
end

if nargin < 2
    n = K * 100;
end

%% main

% generate data

d = 2;
X = randn(d, n);

% perform K-means

[~, s] = kmedoid(X, K, 'display', 'iter');

% visualize results

figure;
plot(X(1,:), X(2,:), 'g.', 'MarkerSize', 6);

hold on;
plot(X(1, s), X(2, s), 'r+', 'LineWidth', 2, 'MarkerSize', 15);

title('K-medoid Demo');

