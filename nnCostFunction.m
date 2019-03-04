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
a= zeros(m,hidden_layer_size);
 X = [ones(m,1) X];
 a = sigmoid(X*transpose(Theta1));
 a = [ones(m,1) a];
 h = sigmoid(a*transpose(Theta2));
 y1 = zeros(m,num_labels);
 q = zeros(m,num_labels);
 for i = 1:m
   for j = 1:num_labels
     if (y(i,1)==j)
       y1(i,j)=1;
     endif
     
   endfor
 endfor
 
 q = (y1.*log(h)+(1-y1).*log(1-h));
 s=0;
 for i = 1:m
   for j = 1:num_labels
     s = s + q(i,j);
   endfor
 endfor

 t1=Theta1.*Theta1;
 t2=Theta2.*Theta2;
 s1=0;
 s2=0;
 
 t1(:,1)=0;
 t2(:,1)=0;
for i = 1:(hidden_layer_size)
   for j = 1:(input_layer_size+1)
     s1 = s1 + t1(i,j);
     
   endfor
 endfor

for i = 1:(num_labels)
   for j = 1:(hidden_layer_size+1)
     s2 = s2 + t2(i,j);
     
   endfor
 endfor 
 tf=0;
 tf = (lambda*(s1+s2))/(2*m);

 
 J=-s/m+tf;



regt1 = Theta1;
regt2 = Theta2;
regt1(:,1)=0;
regt2(:,1)=0;

part_d2 = h-y1;
part_d1temp = (part_d2*Theta2).*(a.*(1-a));

  for i =1:m
    for j =1:hidden_layer_size
      part_d1(i,j)=part_d1temp(i,j+1);
    endfor
  endfor
Theta2_grad = (transpose(part_d2)*a)/m + (lambda/m)*regt2;
Theta1_grad = (transpose(part_d1)*X)/m + (lambda/m)*regt1;
  
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
