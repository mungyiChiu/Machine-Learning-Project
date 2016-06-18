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
% Ytrain = genders_train;
Yactual1 = (genders_train-0.5)*2;

clear images_train;
clear image_features_train;
clear words_train;
clear genders_train;

% Xtrain = [Words_train,image_features_train];
% Xtest = [Words_test,image_features_test];
X = [Xtrain;Xtest];
[m, n] = size(Xtrain);
%X1 = X-repmat(mean(X),size(X,1),1);
[COEFF,SCORE,latent] = pca(X);%pca?????
%cumsum(latent)./sum(latent)
Xcentered = SCORE(:,1:3000) * COEFF(:,1:3000)';%????
%Xcenmean = SCORE + repmat(mean(X),size(SCORE,1),1);
Xcenmean = Xcentered + repmat(mean(X),size(Xcentered,1),1);%??????
Xtrainnew = Xcenmean(1:m,1:3000);
Xtestnew = Xcenmean(m+1:end,1:3000);
%save('Xtrainnew.mat','Xtrainnew');


MD = fitcnb(Xtrainnew,Yactual1);
% save NB.mat MD
cpre = predict(MD,Xtestnew)/2+0.5;
cc = 1 - mean(cpre ~= Ytest)
