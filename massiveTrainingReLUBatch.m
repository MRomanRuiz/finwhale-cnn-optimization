function [statistics, results] = massiveTrainingReLUBatch(trainingData,...
    testData,num)
% Function that performs neural network training while varying
% different parameters to study its performance. Input parameters:
% training image set, test image set,
% and number of times to repeat the training.
% Returns: result statistics and information of that table.

rng(42);
% More comprehensive data augmentation
imageAugmenter = imageDataAugmenter( ...
    'RandRotation', [-15 15], ...            % Random rotations between -15° and 15°
    'RandXTranslation', [-3 3], ...          % Horizontal translation
    'RandYTranslation', [-3 3], ...          % Vertical translation
    'RandXReflection', true, ...             % Random horizontal reflection
    'RandYReflection', true, ...             % Random vertical reflection
    'RandScale', [0.9 1.1] ...               % Random scaling from 90% to 110%
);
[trainData, valData] = splitEachLabel(trainingData, 0.8, 'randomized');
trainData = augmentedImageDatastore([21 22], trainData, 'DataAugmentation', imageAugmenter);

% Network training options
options = trainingOptions('adam', ...
  'MiniBatchSize', 16, ...
  'InitialLearnRate', 1e-3, ...
  "LearnRateSchedule","piecewise",...
  'ValidationFrequency',5, ...
  'ValidationData',valData,...
  "ValidationPatience",5,...
  'MaxEpochs', 10);

statistics = zeros(num,1);
results(num) = struct('accuracy', [], 'precision', [], 'recall', [], 'F1', [], 'confMatrix', []);
% Perform the training and testing loop
for k = 1:num
    fprintf('Iteration %d\n',k)
    % Define the network layers
    layers = [
        imageInputLayer([21 22 1])
        convolution2dLayer(5,64,'Padding','same')
        % batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(3)
        fullyConnectedLayer(2)
        softmaxLayer
        classificationLayer 
        ];
    % Train the network
    trainedDetector = trainNetwork(trainData,layers,options);
    % Test the result
    Ytest = classify(trainedDetector,testData);
    dif = Ytest == testData.Labels;
    eficacia = sum(dif)/length(dif);
    % Save the results
    statistics(k,1) = eficacia;
    % Confusion matrix
    C = confusionmat(testData.Labels, Ytest);
    % disp('Confusion matrix:');
    % disp(C);
    
    % Calculate precision, recall and F1-score
    % Assuming two classes: positive and negative
    TP = C(2,2); FP = C(1,2); FN = C(2,1); TN = C(1,1);
    accuracy = (TP + TN)/(TP + TN + FP + FN);
    precision = TP/(TP+FP);
    recall = TP/(TP+FN);
    F1 = 2*(precision*recall)/(precision+recall);

    results(k).accuracy = accuracy;
    results(k).precision = precision;
    results(k).recall = recall;
    results(k).F1 = F1;
    results(k).confMatrix = C;
end
end
