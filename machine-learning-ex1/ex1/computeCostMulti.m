function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% J_theta = sum(h_theta(i) - y(i))/ 2m, where i = 1 to m and
% h_theta = X(i, :) * theta

for i = 1:m
    h_theta = X(i, :) * theta;
    J = J + (h_theta - y(i))^2;
end

J = J / (2 * m);

% =========================================================================

end
