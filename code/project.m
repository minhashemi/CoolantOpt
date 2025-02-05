clc; clear; close all;

%% network
N = 6;  

% Adjacency matrix (1 if connected)
A = [ 0  1  1  0  0  0;
      1  0  1  1  0  0;
      1  1  0  1  1  0;
      0  1  1  0  1  1;
      0  0  1  1  0  1;
      0  0  0  1  1  0];

% Resistance matrix (ensures connectivity)
R = [  1  2  3  0  0  0;
       2  1  4  2  0  0;
       3  4  1  3  5  0;
       0  2  3  1  4  6;
       0  0  5  4  1  2;
       0  0  0  6  2  1];

% Node types
node_type = ["Source", "Consumer", "Storage", "Consumer", "Storage", "Sink"];

% total inflow = total outflow
D = [-15; 7; 0; 5; 0; 3];

%% init vars
x = rand(N,1) * 5;  % Initial flow
H = eye(N);         % Approximate Hessian
max_iter = 100;     
tol = 1e-6;        

% Track progress
cost_history = zeros(max_iter,1);
grad_norm_history = zeros(max_iter,1);

% figures
figure(1);
subplot(2,1,1);
cost_plot = plot(NaN, NaN, '-o', 'LineWidth', 1.5);
xlabel('Iteration'); ylabel('Cost'); title('Cost Evolution'); grid on; hold on;

subplot(2,1,2);
grad_plot = plot(NaN, NaN, '-s', 'LineWidth', 1.5);
xlabel('Iteration'); ylabel('Gradient Norm'); title('Gradient Convergence'); grid on; hold on;

% Graph Setup
figure(2);
G = graph(A);
p = plot(G, 'Layout', 'force', 'EdgeLabel', {});
title('Cooling System Network (Updated Flow)');
hold on;

% Get edge indices
[s, t] = find(tril(A));  % avoid duplicates
num_edges = length(s);
edge_flows = zeros(num_edges,1);  % Store flows for edges

%% Objective Function
function cost = objective(x, R)
    cost = sum((R * x) .^ 2);
end

%% Gradient Function
function g = gradient(x, R)
    g = 2 * (R' * (R * x));
end

%% BFGS quasi-newton
for iter = 1:max_iter
    g = gradient(x, R);  
    
    
    cost_history(iter) = objective(x, R);
    grad_norm_history(iter) = norm(g);
    
    % Check convergence
    if norm(g) < tol
        disp(['Converged at iteration ', num2str(iter)]);
        cost_history = cost_history(1:iter);
        grad_norm_history = grad_norm_history(1:iter);
        break;
    end
    
    % find search direction
    p_dir = -H * g;
    
    % Line search 
    alpha = 0.1;
    
    % Update flow 
    x_new = x + alpha * p_dir;
    
    % find gradient difference
    g_new = gradient(x_new, R);
    s_vec = x_new - x;
    y_vec = g_new - g;
    
    % BFGS Hessian update
    rho = 1 / (y_vec' * s_vec);
    if rho > 0  % positive-definiteness
        H = (eye(N) - rho * (s_vec * y_vec')) * H * (eye(N) - rho * (y_vec * s_vec')) + rho * (s_vec * s_vec');
    end
    
    % Update vars
    x = x_new;
    
    % Display Progress
    fprintf('Iteration %d | Cost: %.5f | Gradient Norm: %.4f\n', iter, cost_history(iter), grad_norm_history(iter));
    
    % Update Plots
    set(cost_plot, 'XData', 1:iter, 'YData', cost_history(1:iter));
    set(grad_plot, 'XData', 1:iter, 'YData', grad_norm_history(1:iter));
    drawnow;
    
    % Update Network 
    for e = 1:num_edges
        edge_flows(e) = abs(x(s(e)) - x(t(e)));  % Flow difference between nodes
    end
    p.EdgeLabel = arrayfun(@(val) sprintf('%.2f', val), edge_flows, 'UniformOutput', false);
    p.LineWidth = 2 * edge_flows / max(edge_flows);  % Scale line thickness
    drawnow;
    
    pause(0.2);
end

%% Results
disp('Optimal Flow Distribution:');
for i = 1:N
    fprintf('Node %d (%s): Flow = %.4f\n', i, node_type(i), x(i));
end
disp(['Final Minimum Pumping Cost: ', num2str(objective(x, R))]);