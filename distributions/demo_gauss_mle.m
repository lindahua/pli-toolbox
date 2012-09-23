function demo_gauss_mle()
%DEMO_GAUSS_MLE A simple program to demonstrate the MLE of Gauss models
%
%   DEMO_GAUSS_MLE
%

%% Experiment configurations

n = 2000;

%% Generate data

T = randn(2, 2);
mu0 = randn(2, 1) * 5;
C0 = T * T';

X = bsxfun(@plus, T * randn(2, n), mu0);

%% Estimate model

% estimate Gaussian distributions with different covariance forms

Gs = gauss_mle(X, [], 's');
Gd = gauss_mle(X, [], 'd');
Gf = gauss_mle(X, [], 'f');

%% Display results

display_result('True model', mu0, C0);
display_result('Gs', Gs.mu, Gs.cov);
display_result('Gd', Gd.mu, Gd.cov);
display_result('Gf', Gf.mu, Gf.cov);


%% Visualize results

figure;
set(gcf, 'Position', [0, 0, 1200, 300]);
movegui(gcf, 'center');

datapt_size = 5;

subplot(131);
plot(X(1,:), X(2,:), 'b.', 'MarkerSize', datapt_size);
show_gauss(Gs);
axis equal;

subplot(132);
plot(X(1,:), X(2,:), 'b.', 'MarkerSize', datapt_size);
show_gauss(Gd);
axis equal;

subplot(133);
plot(X(1,:), X(2,:), 'b.', 'MarkerSize', datapt_size);
show_gauss(Gf);
axis equal;


function display_result(name, mu, C)

fprintf('%s:\n', name);
fprintf('====================\n');
fprintf('mu = \n');
disp(mu);
fprintf('cov = \n');
disp(C);
fprintf('\n');



function show_gauss(G)

npts = 1000;

hold on;
gauss2d_contour(G, 1, npts, 'r-');
hold on;
gauss2d_contour(G, 2, npts, 'm-');

