clc
clear all
close all
Words_train = importdata('words_train.txt');
image_features_train = importdata('image_features_train.txt');
Words_test = importdata('words_test.txt');
image_features_test = importdata('image_features_test.txt');
genders_train = importdata('genders_train.txt');

Xtrain = [Words_train,image_features_train];
Xtest = [Words_test,image_features_test];  
Xtrainnew = Xtrain(1:4000,:);
Xtestnew = Xcenmean(4001:4998,:);
Ytrainnew = genders_train(1:4000);
Ytestnew = genders_train(4001:4998);
MD = fitcsvm(Xtrainnew, Ytrainnew,'KernelFunction','kernel_intersection');
[label,score] = predict(MD,Xtestnew);
sum(label~=Ytestnew)/length(Ytestnew)
% addpath('liblinear');
% [precision] = logistic(Xtrainnew, Ytrainnew, Xtestnew, Ytestnew)
%fileID = fopen('submitlg.txt','w');
%fprintf = fprintf(fileID,'%i\n',label);