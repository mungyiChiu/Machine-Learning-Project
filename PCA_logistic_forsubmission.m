clc
clear all
close all

% Words_train = importdata('../../../train/words_train.txt');
% image_features_train = importdata('../../../train/image_features_train.txt');
% Words_test = importdata('../../../test/words_test.txt');
% image_features_test = importdata('../../../test/image_features_test.txt');
% genders_train = importdata('../../../train/genders_train.txt');
% Images_train = importdata('../../../train/images_train.txt');
% Images_test = importdata('../../../test/images_test.txt');
% 
% Xtrain = [Words_train,image_features_train];
% Xtest = [Words_test,image_features_test];
% X = [Xtrain;Xtest];

load('../../../../Validate/XValidate.mat');
load('../../../../Validate/YValidate.mat');
Xtest = [XValidate(:, 1:5000) XValidate(:,35001:35007)];
Ytest = YValidate;

clear XValidate;
clear YValidate;

images_train = importdata('../../../../train/images_train.txt');
image_features_train = importdata('../../../../train/image_features_train.txt');
words_train = importdata('../../../../train/words_train.txt');
genders_train = importdata('../../../../train/genders_train.txt');
Xtrain = [words_train image_features_train];
Ytrain = genders_train;

clear images_train;
clear image_features_train;
clear words_train;
clear genders_train;

X = [Xtrain;Xtest];

[m, n] = size(Xtrain);

[COEFF,SCORE,latent] = pca(X);
% Xcenmean = SCORE + repmat(mean(X),size(SCORE,1),1);
Xtrain = SCORE(1:m,1:3000);
Xtest = SCORE(m+1:end,1:3000);

[m,n] = size(Xtest);
Ytest = ones(m,1);

addpath('liblinear');
model = train(Ytrain, sparse(Xtrain), ['-s 0', 'col']);
save SEMI.mat model
[predicted_label] = predict(Ytest, sparse(Xtest), model, ['-q', 'col']);
prediction = predicted_label;
Acc = 1 - mean(prediction ~= Ytest)
