close all
clear

load Network.mat trainedDetector dif
dir = '90.tiff';
img = filterImage(imread(dir));

% Activate the layers we want to study: first block
figure
imshow(img,'InitialMagnification', 'fit')
title('Test image')
figure
act1 = activations(trainedDetector,img,'conv_1');
sz = size(act1);
act1 = reshape(act1,[sz(1) sz(2) 1 sz(3)]);
I = imtile(mat2gray(act1),'GridSize',[4 2]);
subplot(131)
imshow(I)
title('conv_1','Interpreter','none')
act1 = activations(trainedDetector,img,'relu_1');
sz = size(act1);
act1 = reshape(act1,[sz(1) sz(2) 1 sz(3)]);
I = imtile(mat2gray(act1),'GridSize',[4 2]);
subplot(132)
imshow(I)
title('relu_1','Interpreter','none')
act1 = activations(trainedDetector,img,'maxpool_1');
sz = size(act1);
act1 = reshape(act1,[sz(1) sz(2) 1 sz(3)]);
I = imtile(mat2gray(act1),'GridSize',[4 2]);
subplot(133)
imshow(I)
title('maxpool_1','Interpreter','none')

% Activate the layers we want to study: second block
figure
act1 = activations(trainedDetector,img,'conv_2');
sz = size(act1);
act1 = reshape(act1,[sz(1) sz(2) 1 sz(3)]);
I = imtile(mat2gray(act1),'GridSize',[4 4]);
subplot(131)
imshow(I)
title('conv_2','Interpreter','none')
act1 = activations(trainedDetector,img,'relu_2');
sz = size(act1);
act1 = reshape(act1,[sz(1) sz(2) 1 sz(3)]);
I = imtile(mat2gray(act1),'GridSize',[4 4]);
subplot(132)
imshow(I)
title('relu_2','Interpreter','none')
act1 = activations(trainedDetector,img,'maxpool_2');
sz = size(act1);
act1 = reshape(act1,[sz(1) sz(2) 1 sz(3)]);
I = imtile(mat2gray(act1),'GridSize',[4 4]);
subplot(133)
imshow(I)
title('maxpool_2','Interpreter','none')

% Activate the layers we want to study: third block
figure
act1 = activations(trainedDetector,img,'conv_3');
sz = size(act1);
act1 = reshape(act1,[sz(1) sz(2) 1 sz(3)]);
I = imtile(mat2gray(act1),'GridSize',[8 4]);
subplot(131)
imshow(I)
title('conv_3','Interpreter','none')
act1 = activations(trainedDetector,img,'relu_3');
sz = size(act1);
act1 = reshape(act1,[sz(1) sz(2) 1 sz(3)]);
I = imtile(mat2gray(act1),'GridSize',[8 4]);
subplot(132)
imshow(I)
title('relu_3','Interpreter','none')
act1 = activations(trainedDetector,img,'maxpool_3');
sz = size(act1);
act1 = reshape(act1,[sz(1) sz(2) 1 sz(3)]);
I = imtile(mat2gray(act1),'GridSize',[8 4]);
subplot(133)
imshow(I)
title('maxpool_3','Interpreter','none')

% Activate the layers we want to study: fourth block
figure
act1 = activations(trainedDetector,img,'conv_4');
sz = size(act1);
act1 = reshape(act1,[sz(1) sz(2) 1 sz(3)]);
I = imtile(mat2gray(act1),'GridSize',[8 4]);
subplot(131)
imshow(I)
title('conv_4','Interpreter','none')
act1 = activations(trainedDetector,img,'relu_4');
sz = size(act1);
act1 = reshape(act1,[sz(1) sz(2) 1 sz(3)]);
I = imtile(mat2gray(act1),'GridSize',[8 4]);
subplot(132)
imshow(I)
title('relu_4','Interpreter','none')
act1 = activations(trainedDetector,img,'avgpool2d');
sz = size(act1);
act1 = reshape(act1,[sz(1) sz(2) 1 sz(3)]);
I = imtile(mat2gray(act1),'GridSize',[8 4]);
subplot(133)
imshow(I)
title('avgpool2d','Interpreter','none')
