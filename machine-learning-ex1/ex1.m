%% Machine Learning Online Class - Exercise 1: Linear Regression

%  This file contains code that helps you get started on the
%  linear exercise.
%
% x refers to the population size in 10,000s
% y refers to the profit in $10,000s
%

%% Initialization
% Clear and Close Figures
clear ; close all; clc

% Load Data
fprintf('Loading data ...\n');
data = load('ex1data1.txt');
X = data(:, 1); y = data(:, 2);

m = length(y); % number of training examples

%% ======================= Part 1: Plotting Data ===================
fprintf('Plotting Data ...\n')


% Plot Data
% ====================== YOUR CODE HERE ======================
% TODO: plot the data points and give the figure axes labels of
%       population and profit.

plot (X, y, 'x');
xlabel('Population size in 10,000s');
ylabel('Profit in $10,000s');


% ============================================================

fprintf('Program paused. Press enter to continue.\n');
pause;
%% =================== Part 2: Gradient descent ===================
fprintf('Running Gradient Descent ...\n')

X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters

% Some gradient descent settings
iterations = 1500;
alpha = 0.01;

% run gradient descent
[theta, J_history] = gradientDescent(X, y, theta, alpha, iterations);

% print theta to screen
fprintf('Theta found by gradient descent: %f %f \n', theta(1), theta(2));


% Plot the linear fit
% ====================== YOUR CODE HERE ======================
% TODO: plot the linear function that best fits the given training data set

H = X*theta;

hold on
plot(X(:,2),H);
hold off

xlabel('Population size in 10,000s');
ylabel('Profit in $10,000s');
legend('Training data', 'Linear regression');

% ============================================================

% Predict values for population sizes of 35,000 and 70,000
predict1 = [1, 3.5] *theta;
fprintf('For population=35000, we expect a profit of %f\n',predict1*10000);
predict2 = [1, 7] * theta;
fprintf('For population=70000, we expect a profit of %f\n',predict2*10000);

fprintf('Program paused. Press enter to continue.\n');
pause;
%% ======== Part 3: Plotting J to iteration index ==============
% plot the cost function value as a function of iteration number
% ====================== YOUR CODE HERE ======================
% TODO: plot the cost function value as a function of iteration number
%

iter = (1:iterations);
plot(iter, J_history);
xlabel('Number of iterations');
ylabel('J');

% ============================================================
hold off;
fprintf('Program paused. Press enter to continue.\n');
pause;

%% ============= Part 4: Plotting J to parameters =============
fprintf('Visualizing J(theta_0, theta_1) ...\n')

% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];    
	  J_vals(i,j) = computeCost(X, y, t);
    end
end

% Because of the way meshgrids work in the surf command, we need to 
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
