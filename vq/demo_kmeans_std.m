function demo_kmeans_std(K, n)
%DEMO_KMEANS_STD A simple program to demonstrate the use of kmeans_std
%
%   DEMO_KMEANS_STD;
%   DEMO_KMEANS_STD(K);
%   DEMO_KMEANS_STD(K, n);
%
%       Perform K-means on n randomly generated points.
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

C = kmeans_std(X, K, 'display', 'iter');

% visualize results

figure;
plot(X(1,:), X(2,:), 'g.', 'MarkerSize', 6);

hold on;
plot(C(1,:), C(2,:), 'r+', 'LineWidth', 2, 'MarkerSize', 15);

title('K-means Demo');

