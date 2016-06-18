clc
clear all
close all

Words_train = importdata('../../../train/words_train.txt');
image_features_train = importdata('../../../train/image_features_train.txt');
Words_test = importdata('../../../test/words_test.txt');
image_features_test = importdata('../../../test/image_features_test.txt');
genders_train = importdata('../../../train/genders_train.txt');
Images_train = importdata('../../../train/images_train.txt');
Images_test = importdata('../../../test/images_test.txt');

Xtrain = [Words_train,image_features_train];
Ytrain = genders_train;
Xtest = [Words_test,image_features_test];  
MD = fitcknn(Xtrain,Ytrain,'NumNeighbors',30,'Standardize',1);
save KNN.mat MD 
[label,score] = predict(MD,Xtest);
prediction = label;