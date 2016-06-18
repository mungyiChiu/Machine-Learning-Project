function predictions = make_final_prediction(model, XTest, XTrain)
% Input
% X_test : a nxp vector representing "n" test samples with p features.
% X_test=[words images image_features] a n-by-35007 vector
% model : struct, what you initialized from init_model.m
%
% Output
% prediction : a nx1 which is your prediction of the test samples

% Sample model
addpath ('./libsvm');
% load('Xtrain.mat');
[m,n] = size(XTest);
% load('FS2.mat')
Ktest = pdist2([XTrain(:, 1:5000) XTrain(:, 35001:35007)], [XTest(:, 1:5000) XTest(:, 35001:35007)], @(x, Y) sum(bsxfun(@min, x, Y), 2))';
Ytest = ones(m,1);
[labels acc vals] = svmpredict(Ytest, [(1:size(Ktest,1))' Ktest], model);
predictions = labels;


