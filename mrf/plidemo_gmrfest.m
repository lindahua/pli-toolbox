function plidemo_gmrfest()
%PLIDEMO_GMRFEST Demos the use of pli_gmrfest in GMRF estimation
%
%   PLIDEMO_GMRFEST;
%
%       Demos the estimation of an MRF in 4-connected spatial grid.
%

%% configuration

h = 8;
w = 10;
n = 5000;

lambda = 10;

%% generate data

[x, y] = meshgrid(1:w, 1:h);
x = x(:);
y = y(:);

dx = abs(bsxfun(@minus, x, x.'));
dy = abs(bsxfun(@minus, y, y.'));

pat = zeros(h*w, h*w);
pat(dx == 0 & dy == 0) = 1;
pat(dx == 1 & dy == 0) = 2;
pat(dx == 0 & dy == 1) = 3;

phi_gt = [12 -2 -3]';

A_gt = full(pli_pat2mat(pat, phi_gt));
C_gt = inv(A_gt);
C_gt = 0.5 * (C_gt + C_gt');

G = pli_makegauss(h*w, 0, C_gt);
X = pli_gauss_sample(G, n);


%% estimation

phi0 = [1 0 0].';
opts = pli_optimset('Display', 'iter', 'tolfun', 1.0e-14);

[phi, objv] = pli_gmrfest(pat, X, lambda, phi0, opts);

A = pli_pat2mat(pat, phi);
objv2 = eval_objv(phi, A, X, lambda);
fprintf('objective = %.5g (dev = %g)\n', objv, abs(objv - objv2));

disp('Results [phi_gt phi]:');
disp([phi_gt phi]);
fprintf('relative error energy: %.4g\n\n', ...
    norm(phi - phi_gt)^2 / norm(phi_gt)^2);



function v = eval_objv(phi, A, X, lambda)

d = size(A, 1);
L = (-0.5) * (sum(X .* (A * X), 1) - pli_logdet(A) + d * log(2*pi));

if isscalar(lambda)
    lambda = lambda * ones(size(phi));
end
v = - sum(L) + 0.5 * (lambda' * (phi).^2);
