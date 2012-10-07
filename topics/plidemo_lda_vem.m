function plidemo_lda_vem()
%PLIDEMO_LDA_VEM Demonstrate the use of LDA with variational EM
%
%   PLIDEMO_LDA_VEM;
%

%% Experiment configuration

V = 8;
K = 3;
doc_len = 200;
ndocs = 100;
ndocs_show = 5;

%% Generate data

disp('Underlying model');
disp('========================');

% topics

U = exp(8 * rand(V, K));
U = bsxfun(@times, U, 1 ./ sum(U, 1));

for k = 1 : K
    print_vec(sprintf('Topic [%d]', k), '%.4f ', U(:,k));
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
    p = U * theta;    
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

vi_problem = pli_lda_vinfer_problem(alpha, U);

for i = 1 : ndocs_show
    
    h = H(:, i);
    sol = vi_problem.init_solution(h);
    
    sol = pli_iteroptim('maximize', @vi_problem.eval_objv, ...
        @vi_problem.update, sol, ...
        'display', 'off');
        
    print_vec(sprintf('Doc [%d].gamma', i), '%8.4f ', sol.gamma); 
    print_vec('        pmean', '%8.4f ', sol.gamma / sum(sol.gamma));
    print_vec('        theta', '%8.4f ', Theta(:,i));
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

vem_problem = pli_lda_vem_problem(V, 0.01);
vem_problem.set_docs(H);

U_init = U + 0.1 * rand(size(U));
U_init = bsxfun(@times, U_init, 1 ./ sum(U, 1));

sol = vem_problem.init_solution(U_init);

sol = pli_iteroptim('maximize', @vem_problem.eval_objv, ...
    @vem_problem.update, sol, ...
    'maxiter', 100, ...
    'display', 'iter');

print_vec('est  alpha', '%7.4f ', sol.alpha);
print_vec('true alpha', '%7.4f ', alpha);


function print_vec(title, spec, v)

fprintf('%s: ', title);

n = numel(v);
for j = 1 : n
    fprintf(spec, v(j));
end

fprintf('\n');




