%% Massive training tests for obtaining statistics
%% Single convolutional layer 
% Training
clear
clc
% Training images
dir = 'TrainingCrops';
train_labels = categorical([repelem("pulse",1232) repelem("negative",1428)]');
train = imageDatastore(dir,"IncludeSubfolders",true,Labels=train_labels);
train = shuffle(train);
% Test images
dir = 'TestCrops';
test_labels = categorical([repelem("pulse",286) repelem("negative",294)]');
test = imageDatastore(dir,"IncludeSubfolders",true,Labels=test_labels);
test = shuffle(test);
% Ranges
ranges = [2,10,3,6];
reps = 10;
% Application of our function
[statistics, results, info] = massiveTrainingConv(train,test,ranges,reps);
% Clean up the workspace
clearvars dir test_labels train_labels ranges reps

%% Single convolutional layer 
% Analysis of the results
close all
clear
clc
% Load data
load testsConv.mat statistics info
% Extract information for graph labels
sizes = string.empty;
for i = 1:size(info,2)
    division = split(info(1,i)," to ");
    sizes(i) = division(2);
end
sizes = categorical(sizes);
reps = string.empty;
for i = 1:size(info,1)
    division = split(info(i,1)," to ");
    reps(i) = division(1);
end
reps = categorical(reps);
% Surface representation
[~,~,n] = size(statistics);
colores = rand(n,3);
figure
hold on
for i = 1:n
    nombre = "Iteration " + int2str(i);
    surf(statistics(:,:,i),'FaceColor',colores(i,:),...
        'DisplayName',nombre)
end
xticklabels(sizes);
yticks([1,2,3,4])
yticklabels(reps);
grid
xlabel("Filter size")
ylabel("Number of filters")
zlabel("Accuracy")
legend('Location','southwest')
title("Deep Neural Network: one convolutional layer")
hold off

view(0,0)
% view(15,25)

% Analyze the results 
% Find the best training performed
[maximo, lineal_max] = max(statistics(:));
[fila, columna, pagina] = ind2sub(size(statistics), lineal_max);
caracteristicas = info(fila,columna);
fprintf(['The best result was in iteration %d with a value ' ...
    'of %0.4f\n and the training characteristics were: %s\n'], ...
    pagina, maximo, caracteristicas);
% Find the best average configuration
means = mean(statistics,3);
[maximo,lineal_max] = max(means(:));
[fila, columna, pagina] = ind2sub(size(statistics), lineal_max);
caracteristicas = info(fila,columna);
fprintf(['The best average configuration was in iteration %d with a value ' ...
    'of %0.4f\n and the training characteristics were: %s\n'], ...
    pagina, maximo, caracteristicas);
% Find the mean standard deviation
stds = std(statistics,0,3);

%% Best convolution with variable max pooling
% Training
clear
clc
% Training images
dir = 'TrainingCrops';
train_labels = categorical([repelem("pulse",1232) repelem("negative",1428)]');
train = imageDatastore(dir,"IncludeSubfolders",true,Labels=train_labels);
train = shuffle(train);  
% Test images
dir = 'TestCrops';
test_labels = categorical([repelem("pulse",286) repelem("negative",294)]');
test = imageDatastore(dir,"IncludeSubfolders",true,Labels=test_labels);
test = shuffle(test);
% Ranges
ranges = [1,6];
reps = 10;
% Application of our function
[statistics, results, info] = massiveTrainingMaxPool(train,test,ranges,reps);
% Clean up the workspace
clearvars dir test_labels train_labels ranges reps

%% Best convolution with variable max pooling
% Analysis of the results
close all
clear
clc
% Load data
load testsMaxPool statistics info
% Extract information for graph labels
sizes = string.empty;
for i = 1:size(info,1)
    division = split(info(i,1)," a ");
    sizes(i) = division(2);
end
sizes = categorical(sizes);
% Surface representation
[~,~,n] = size(statistics);
figure
hold on
for i = 1:n
    nombre = "Iteration " + int2str(i);
    plot(statistics(:,1,i),"LineWidth",1,"Color",...
        [rand(),rand(),rand()],"DisplayName",nombre)
end
xticklabels(sizes);
grid
xlabel("Pool size")
ylabel("Accuracy")
legend('Location','southwest')
title("Deep Neural Network: one convolutional layer with maxPooling")
hold off

% Analyze the results 
% Find the best training performed
[maximo, lineal_max] = max(statistics(:));
[fila, columna, pagina] = ind2sub(size(statistics), lineal_max);
caracteristicas = info(fila,columna);
fprintf(['The best result was in iteration %d with a value ' ...
    'of %0.4f\n and the training characteristics were: %s\n'], ...
    pagina, maximo, caracteristicas);
% Find the best average configuration
means = mean(statistics,3);
[maximo,lineal_max] = max(means(:));
[fila, columna] = ind2sub(size(means), lineal_max); 
caracteristicas = info(fila,columna);
fprintf(['The best average configuration had a value of ' ...
    '%0.4f\n and the training characteristics were: %s\n'], ...
    maximo, caracteristicas);
% Find the mean standard deviation
stds = std(statistics,0,3);

%% Best convolution with relu and/or batch and best max pooling
% Training
clear
clc
% Training images
dir = 'TrainingCrops';
train_labels = categorical([repelem("pulse",1232) repelem("negative",1428)]');
train = imageDatastore(dir,"IncludeSubfolders",true,Labels=train_labels);
train = shuffle(train);
% Test images
dir = 'TestCrops';
test_labels = categorical([repelem("pulse",286) repelem("negative",294)]');
test = imageDatastore(dir,"IncludeSubfolders",true,Labels=test_labels);
test = shuffle(test);
% Ranges 
reps = 10;
% Application of our function
[statistics,results] = massiveTrainingReLUBatch(train,test,reps);
% Clean up the workspace
clearvars dir test_labels train_labels ranges reps

%% Best convolution with relu and/or batch and best max pooling
% Analysis of the results
close all
clear
clc
% Load data
load testsReLU statistics
% load testsBatch statistics
% Surface representation
plot(statistics,"LineWidth",1)
grid
xlabel("Iteration")
ylabel("Accuracy")
title("Deep Neural Network: one convolutional layer with best maxPooling and ReLu")
hold off

% Analyze the results 
% Find the best training performed
[maximo, lineal_max] = max(statistics);
fprintf(['The best result was in iteration %d with a value ' ...
    'of %0.4f\n'], lineal_max, maximo);
% Find the best average configuration
mean = mean(statistics);
fprintf('The average result was %0.4f\n', mean);
% Find the mean standard deviation
stds = std(statistics);
fprintf('The mean deviation is %0.4f\n', stds);

%% Network designed
% Training
clear
clc
% Training images
dir = 'TrainingCrops';
train_labels = categorical([repelem("pulse",1232) repelem("negative",1428)]');
train = imageDatastore(dir,"IncludeSubfolders",true,Labels=train_labels);
train = shuffle(train);
% Test images
dir = 'TestCrops';
test_labels = categorical([repelem("pulse",286) repelem("negative",294)]');
test = imageDatastore(dir,"IncludeSubfolders",true,Labels=test_labels);
test = shuffle(test);
% Ranges
reps = 10;
% Application of our function
[statistics,results] = massiveTrainingNetwork(train,test,reps);
% Clean up the workspace
clearvars dir test_labels train_labels ranges reps

%% Network designed
% Analysis of the results
close all
% clear
% clc
% Load data
load NstatisticsPaper.mat statistics
% Surface representation
plot(statistics,"LineWidth",1)
grid
xlabel("Iteration")
ylabel("Accuracy")
title("Deep Neural Network: detection network")
hold off

% Analyze the results 
% Find the best training performed
[maximo, lineal_max] = max(statistics);
fprintf(['The best result was in iteration %d with a value ' ...
    'of %0.4f\n'], lineal_max, maximo);
% Find the best average configuration
mean = mean(statistics);
fprintf('The average result was %0.4f\n', mean);

%% Network designed: only three blocks
% Analysis of the results
close all
clear
clc
% Load data
load NstatisticsPaperA3.mat statistics
% Surface representation
plot(statistics,"LineWidth",1)
grid
xlabel("Iteration")
ylabel("Accuracy")
title("Deep Neural Network: detection network (3)")
hold off

% Analyze the results 
% Find the best training performed
[maximo, lineal_max] = max(statistics);
fprintf(['The best result was in iteration %d with a value ' ...
    'of %0.4f\n'], lineal_max, maximo);
% Find the best average configuration
mean = mean(statistics);
fprintf('The average result was %0.4f\n', mean);
