function plidemo_kmedoid(K, n, op)
%PLIDEMO_KMEDOID A simple program to demonstrate the use of kmedoid
%
%   PLIDEMO_KMEDOID;
%   PLIDEMO_KMEDOID(K);
%   PLIDEMO_KMEDOID(K, n);
%
%       Perform K-medoid on n randomly generated points.
%
%   PLIDEMO_KMEDOID(K, n, 'c');
%
%       Perform K-medoid using pre-computed cost matrix.
%
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

if nargin < 3
    use_c = 0;
else
    if ~strcmpi(op, 'c')
        error('plidemo_kmedoid:invalidarg', 'The 3rd argument is invalid.');
    end
    use_c = 1;
end


%% main

% generate data

d = 2;
X = randn(d, n);

% perform K-means

if use_c
    C = pli_pw_euclidean(X, [], 'sq');
    [~, s] = pli_kmedoid_c(C, K, 'display', 'iter');    
else
    [~, s] = pli_kmedoid(X, K, 'display', 'iter');
end

% visualize results

figure;
plot(X(1,:), X(2,:), 'g.', 'MarkerSize', 6);

hold on;
plot(X(1, s), X(2, s), 'r+', 'LineWidth', 2, 'MarkerSize', 15);

title('K-medoid Demo');

