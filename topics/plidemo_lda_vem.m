function plidemo_lda_vem()
%PLIDEMO_LDA_VEM Demonstrate the use of LDA with variational EM
%
%   PLIDEMO_LDA_VEM;
%

%% Experiment configuration

V = 8;
K = 3;
doc_len = 1000;
ndocs = 100;
ndocs_show = 5;

%% Generate data

disp('Underlying model');
disp('========================');

% topics

U = exp(8 * rand(K, V));
U = bsxfun(@times, U, 1 ./ sum(U, 2));
logU = log(U);

for k = 1 : K
    print_vec(sprintf('Topic [%d]', k), '%.4f ', U(k,:));
end

disp(' ');

% Dirichlet prior

alpha = 1.2 + rand(K, 1) * 3.0;
print_vec('alpha', '%.4f ', alpha);

disp(' ');

% documents

disp('Documents');
disp('========================');

Theta = pli_dirichlet_sample(alpha, ndocs);

H = zeros(V, ndocs);
for i = 1 : ndocs
    
    theta = Theta(:, i);
    p = theta' * U;
    h = mnrnd(doc_len, p);
    
    H(:,i) = h;
end

for i = 1 : ndocs_show
    print_vec(sprintf('Doc [%d].hist', i), '%3d ', H(:,i));
end

disp(' ');


%% Inference for individual documents

disp('Document inference');
disp('  gamma: posterior param');
disp('  pmean: posterior mean');
disp('  theta: underlying p');
disp('========================');

for i = 1 : ndocs_show
    
    h = H(:, i);    
    gam = pli_lda_vinfer(alpha, logU, h, 'display', 'off');
    
    theta = Theta(:,i);
    p = gam / sum(gam);
    
    print_vec(sprintf('Doc [%d].gamma', i), '%8.4f ', gam); 
    print_vec('        pmean', '%8.4f ', p);
    print_vec('        theta', '%8.4f ', theta);
      fprintf('        L1-dev = %.4f\n', norm(theta - p, 1));
    fprintf('\n');
end

disp(' ');

%% Estimation using Variational EM

answer = input('Do you want to see Variational EM ? (y/n): ', 's');
if ~strcmpi(answer, 'y')
    return;
end

disp('Model Estimation (VEM)');
disp('========================');

U_init = U + 0.1 * rand(size(U));
U_init = bsxfun(@times, U_init, 1 ./ sum(U, 1));

sol = pli_lda_vem(H, U_init, 'pricnt', 1, 'display', 'iter', 'maxiter', 100);

print_vec('est  alpha', '%7.4f ', sol.alpha);
print_vec('true alpha', '%7.4f ', alpha);

fprintf('\nEstimated topics:\n');
for k = 1 : K
    print_vec(sprintf('Topic [%d] gtr', k), '%.4f ', sol.U(k,:));
    print_vec('          est', '%.4f ', U(k,:));
end

disp(' ');


function print_vec(title, spec, v)

fprintf('%s: ', title);

n = numel(v);
for j = 1 : n
    fprintf(spec, v(j));
end

fprintf('\n');




