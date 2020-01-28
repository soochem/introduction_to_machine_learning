%% Machine Learning Online Class

%  This file contains code that helps you get started on the
%  linear regression exercise. 

%% Initialization
% Clear and Close Figures
clear ; close all; clc

% Load Data
fprintf('Loading data ...\n');
data = load('ex1data2.txt');
X = data(:, 1:2); y = data(:, 3);

m = length(y); % number of training examples

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');
mu = mean(X);
sigma = std(X(:,1));
X = (X-mu)/sigma;

% Add intercept term to X
X = [ones(m, 1) X];

%% ================ Part 5: Gradient Descent ================
fprintf('Running gradient descent ...\n');

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, 0.01, 5000);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 br house
x_new = [1650 3];
x_new = (x_new-mu)/sigma; % Scale features and set them to zero mean
x_new = [1 x_new];
price = x_new*theta;
fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%.2f\n'], price);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 6: Normal Equations ================
fprintf('Solving with normal equations...\n');

% Load Data
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');


% Estimate the price of a 1650 sq-ft, 3 br house
x_new = [1650 3];
x_new = [1 x_new];
price = x_new*theta;
fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price);

