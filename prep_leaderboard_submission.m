% Load data
img_test = importdata('../test/images_test.txt');
img_feat_test = importdata('../test/image_features_test.txt');
word_test = importdata('../test/words_test.txt');
X_test = [word_test img_test img_feat_test];

tm = tic;
model = init_model;
toc(tm) < 180

images_train = importdata('../train/images_train.txt');
image_features_train = importdata('../train/image_features_train.txt');
words_train = importdata('../train/words_train.txt');
Xtrain = [words_train images_train image_features_train];

tp = tic;
predictions = make_final_prediction(model, X_test, Xtrain);
toc(tp) < 600
% Use turnin on the output file
% turnin -c cis520 -p leaderboard submit.txt
dlmwrite('submit.txt', predictions);