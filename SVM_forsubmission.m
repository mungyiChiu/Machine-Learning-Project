clc
clear all
close all

addpath ('./libsvm');
Words_train = importdata('../../../train/words_train.txt');
image_features_train = importdata('../../../train/image_features_train.txt');
Words_test = importdata('../../../test/words_test.txt');
image_features_test = importdata('../../../test/image_features_test.txt');
genders_train = importdata('../../../train/genders_train.txt');
Images_train = importdata('../../../train/images_train.txt');
Images_test = importdata('../../../test/images_test.txt');

Y = genders_train; 
X = [Words_train, image_features_train];
Xtest = [Words_test, image_features_test];
[m,n] = size(Xtest);
Ytest = ones(m,1);

K = pdist2(X, X, @(x, Y) sum(bsxfun(@min, x, Y), 2))';
Ktest = pdist2(X, Xtest, @(x, Y) sum(bsxfun(@min, x, Y), 2))';

crange = 10.^[-10:0.1:3];
parfor i = 1:numel(crange)
    acc(i) = svmtrain(Y, [(1:size(K,1))' K], sprintf('-t 4 -v 10 -c %g', crange(i)));
end
[~, bestc] = max(acc);
fprintf('Cross-val chose best C = %g\n', crange(bestc));
model = svmtrain(Y, [(1:size(K,1))' K], sprintf('-t 4 -c %g', crange(bestc)));
save SVM.mat model
[yhat acc vals] = svmpredict(Ytest, [(1:size(Ktest,1))' Ktest], model);
prediction = yhat; 
