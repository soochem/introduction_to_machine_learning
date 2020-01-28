function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples
J = 0; % variable for storing cost function value 

% ====================== YOUR CODE HERE ===================================
% TODO: Compute the cost of a particular choice of theta
%

S = sum(((X*theta) - y) .^2);
J = S/(2*m);
    
% =========================================================================

end
