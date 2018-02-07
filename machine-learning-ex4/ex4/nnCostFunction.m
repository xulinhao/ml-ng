function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

% make y's result vector Y, where Y = 5000 x 10
I = eye(num_labels);
Y = zeros(m, num_labels);
for i = 1:m
  Y(i, :) = I(y(i), :);
end

% compute vector h_theta, where h_theta = 5000 x 10
% A1 = 5000 x 401
A1 = [ones(m, 1) X];

% A2 = 5000 x 26
Z2 = A1 * Theta1';
A2 = [ones(m, 1) sigmoid(Z2)];

% A3 = 5000 x 10
Z3 = A2 * Theta2';
A3 = sigmoid(Z3);

% H(theta) = 5000 x 10
H_theta = A3;

J_org = sum(sum(-Y .* log(H_theta) - (1 - Y) .* log(1 - H_theta))) / m;
J_reg = (lambda * (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2)))) / (2 * m);

J = J_org + J_reg;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

% Step 1: see the implementation in Part 1: A1, Z2, A2, Z3, A3
% Step 2: D3 = 5000 x 10
D3 = A3 - Y;

% Step 3: D2 = 5000 x 26
D2 = D3 * Theta2 .* sigmoidGradient([ones(m, 1) Z2]);

% Step 4: D2 = 5000 x 25 (ignoring D2(0))
D2 = D2(:, 2:end);

DL_2 = D3' * A2; % 10 x 26
DL_1 = D2' * A1; % 25 x 401

% Step 5: no regularization
Theta2_grad = DL_2 / m;
Theta1_grad = DL_1 / m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta2_temp = [zeros(size(Theta2), 1) Theta2(:, 2:end)];
Theta1_temp = [zeros(size(Theta1), 1) Theta1(:, 2:end)];

Theta2_grad = Theta2_grad + lambda * Theta2_temp / m;
Theta1_grad = Theta1_grad + lambda * Theta1_temp / m;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
