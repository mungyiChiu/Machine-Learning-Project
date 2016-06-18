function [ precision, predicted_label] = logistic( train_x, train_y, test_x, test_y )
    model = train(train_y, sparse(train_x), ['-s 0', 'col']);
    [predicted_label] = predict(test_y, sparse(test_x), model, ['-q', 'col']);
    precision = 1 - sum(predicted_label~=test_y) / length(test_y);
end

