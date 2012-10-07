function plidemo_gmm_em()
%PLIDEMO_GMM_EM Demonstartes GMM estimation using EM
%
%   PLIDEMO_GMM_EM();
%

%% Experiment configuration

d = 2;
K = 3;
n = 1000;   % # samples / cluster

inter_sig = 6;
intra_sig = 1;

cform = 'f';

%% Generate data

mu0 = randn(d, K) * inter_sig;
C0 = zeros(d, d, K);

X = cell(1, K);
for k = 1 : K
    Ck = random_cov2d(intra_sig);
    C0(:,:,k) = Ck;
    L = chol(Ck, 'lower');
    X{k} = bsxfun(@plus, L * randn(d, n), mu0(:,k));
end

X = [X{:}];


%% Perform estimation

sol = pli_gmm_em(X, K, 'covform', cform, ...
    'display', 'iter');

G = sol.components;

%% Visualize results

% plot data points

plot(X(1,:), X(2,:), 'b.', 'MarkerSize', 5);

% plot estimated model

npts = 1000;
hold on;
pli_gauss2d_contour(G, 1, npts, 'r-', 'LineWidth', 2);
hold on;
pli_gauss2d_contour(G, 2, npts, 'm-', 'LineWidth', 2);

axis equal;


function C = random_cov2d(r)

t = rand() * (2 * pi);

R = [cos(t) -sin(t); sin(t) cos(t)];

dv = (0.3 + 1.4 * rand(2, 1)) * r;
C = R' * diag(dv) * R;


