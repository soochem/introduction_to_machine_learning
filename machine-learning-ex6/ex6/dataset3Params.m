function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================

epsilon =  10000;
choice = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
for i = 1:length(choice) % C
    for j = 1:length(choice) % sigma
        C1 = choice(i);
        sigma1 = choice(j); 
        model = svmTrain(X, y, C1, @(x1, x2) gaussianKernel(x1, x2, sigma1));
        predict = svmPredict(model, Xval);
        error = mean(double(predict ~= yval));
        if error < epsilon
            C = C1;
            sigma = sigma1;
            epsilon = error;
            %disp(C)
            %disp(sigma)

        end
    end
end


% =========================================================================

end
