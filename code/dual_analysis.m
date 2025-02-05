clc; clear; close all;

project; 

N = size(A,1);

% Hessian matrix of cost function
Q = R' * R;  

% Construct KKT matrix
KKT_matrix = [Q, A'; 
              A, zeros(N)];

% RHS vector
rhs = [zeros(N,1); D];

% Solve KKT 
solution = KKT_matrix \ rhs;

% primal and dual solutions
x_opt = solution(1:N);
lambda_dual = solution(N+1:end);

%% Results
disp('Optimal Flow Distribution:');
disp(x_opt);

disp('Dual Variables (Shadow Prices):');
disp(lambda_dual);

%% Interpretation of Dual Variables
figure;
bar(lambda_dual);
xlabel('Node Index');
ylabel('Shadow Price (Dual Variable)');
title('Dual Variables - Impact of Demand Constraints');
grid on;

% critical nodes
[~, critical_nodes] = maxk(abs(lambda_dual), 3);
disp('Critical Nodes (Highest Impact on Cost):');
disp(critical_nodes);
