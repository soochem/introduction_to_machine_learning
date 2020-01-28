function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with
%regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

H = sigmoid(X*theta);
J = (-1/m)*sum(y'*log(H)+(1-y)'*log(1-H)) + (lambda/(2*m))*(theta(2:end)'*theta(2:end));
%J = (-1/m)*(y'*log(H)+(1-y')*log(1-H)) + (lambda/2*m)*sum(theta(2:end).^2);
%논리형 스칼라 에러, 인덱스 에러 - j가 저장되지 않음

grad(1,1) = (1/m)*(X(:,1))'*(H-y);
grad(2:end,1)=(1/m)*X(:,2:end)'*(H-y) + (lambda/m)*(theta(2:end));
grad = grad(:);

% =============================================================

end
