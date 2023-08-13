% Model training 
clear all
close all
clc
%% Input for Training set 
rootFolder = fullfile('D:\saliency based video summarzation\ICM\train');
categories  = {'real','attack'};
trainingset1 = imageDatastore(fullfile(rootFolder, categories), 'IncludeSubfolders',true, ...
 'LabelSource','foldernames');

%% Input for development set 
rootFolder = fullfile('D:\saliency based video summarzation\ICM\test');
categories  = {'real','attack'};
developmentset1 = imageDatastore(fullfile(rootFolder, categories), 'IncludeSubfolders',true, ...
 'LabelSource','foldernames');

%% Input for Testing set 
rootFolder = fullfile('D:\saliency based video summarzation\Oulu testing set');
categories = {'real','attack'};
testingsetdata1 = imageDatastore(fullfile(rootFolder, categories),  'IncludeSubfolders',true, ...
 'LabelSource','foldernames');

%% Extracting  labels for training, development, and test set
trainingLabels1 = trainingset1.Labels;
developmentlabel1 = developmentset1.Labels;
testinglabel1 = testingsetdata1.Labels;

%% Input for Deep learning model
net = densenet201;

%% Resize images and perform data augmentation for the input of RESnET MODEL
 trainingset1.ReadFcn = @(filename)readAndPreprocessImage(filename);
inputSize = net.Layers(1).InputSize;
 imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-20,20], ...
    'RandXTranslation',[-3 3], ...
    'RandYTranslation',[-3 3]);
 augimdsTrain1 = augmentedImageDatastore(inputSize(1:2),trainingset1, ...
     'DataAugmentation',imageAugmenter);

%% Features extraction based on the last average pooling layer of ResNet for the training set 
featureLayer =  'avg_pool';
trainingFeatures1 = activations(net, augimdsTrain1, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
%% Features extraction based on the last average pooling layer of ResNet for the development set 
 developmentset1.ReadFcn = @(filename)readAndPreprocessImage(filename);
developmentFeatures1 = activations(net, developmentset1, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

%% Features extraction based on the last average pooling layer of ResNet for the testing set 
% Extract testing set features using the CNN
testingsetdata1.ReadFcn = @(filename)readAndPreprocessImage(filename);
testingFeatures1 = activations(net, testingsetdata1, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

%%
% GRU  
% Converting data into RNN FORMAT
rng(0);
trainf = {};
trainf{end+1} =  trainingFeatures1;

trainlabl = {};
trainlabl{end+1} = trainingLabels1';

train1 = {};
train1{end+1} = developmentFeatures1;
% 
train2 = {};
train2{end+1} = developmentlabel1';


numFeatures = 1920;
 numHiddenUnits =500;
numClasses = 2;
layers = [ ...
    sequenceInputLayer(numFeatures)
         gruLayer(numHiddenUnits,'OutputMode','sequence','RecurrentWeightsInitializer','he')
     fullyConnectedLayer(numClasses,'WeightsInitializer','he')
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
     'ExecutionEnvironment','gpu', ... 
       'InitialLearnRate',0.0001, ...
    'MaxEpochs',1500, ...
    'ValidationData',{train1,train2}, ...
    'ValidationFrequency',30, ...
     'SequenceLength','longest', ...
    'Plots','training-progress',...
   'OutputFcn',@(info)stopIfAccuracyNotImproving(info,3));

lstm1 = trainNetwork(trainf',trainlabl,layers,options);

[predictedLabels4, devlp_scores1] = classify(lstm1, developmentFeatures1);
% Converting labels into numerical form
 numericLabels1 = grp2idx(developmentlabel1);
 numericLabels1(numericLabels1==2)= -1;
 numericLabels1(numericLabels1==1)= 1;
 devlpscores1 =devlp_scores1';

 % Evaluation for testing set for EER
 [TPR,TNR,Info]=vl_roc(numericLabels1,devlpscores1(:,1));
 EER = Info.eer*100
 threashold = Info.eerThreshold;

 % Evaluation for testing set for HTER 
 [predictedLabels2, test_scores2] = classify(lstm1, testingFeatures1);
 testscores2 = test_scores2';
 numericLabels = grp2idx(testinglabel1);
 numericLabels(numericLabels==2)= -1;
 numericLabels(numericLabels==1)= 1;
 real_scores1 =  testscores2(numericLabels==1);
 attack_scores2 =  testscores2(numericLabels==-1);
 FAR = sum(attack_scores2>threashold) / numel(attack_scores2)*100;
 FRR = sum(real_scores1<=threashold) / numel(real_scores1)*100;
 HTER1 = (FAR+FRR)/2
 
[x1,y1,threshold,AUC1] = perfcurve(numericLabels, testscores2(:,1),1);
AUC1
plot(x1,y1,'-g','LineWidth',1.8,'MarkerSize',1.8)
grid on
hold on

 %% Input for Training set OF dataset 2
rootFolder = fullfile('D:\saliency based video summarzation\OCM\Trainset');
categories  = {'real','attack'};
trainingset2 = imageDatastore(fullfile(rootFolder, categories), 'IncludeSubfolders',true, ...
 'LabelSource','foldernames');
%% Input for development set 
rootFolder = fullfile('D:\saliency based video summarzation\OCM\Test set');
categories  = {'real','attack'};
developmentset2 = imageDatastore(fullfile(rootFolder, categories), 'IncludeSubfolders',true, ...
 'LabelSource','foldernames');
%% Input for Testing set 
rootFolder = fullfile('D:\saliency based video summarzation\Replay Attack dataset\test');
categories = {'real','attack'};
testingsetdata2 = imageDatastore(fullfile(rootFolder, categories),  'IncludeSubfolders',true, ...
 'LabelSource','foldernames');
%% Extracting  labels for training, development, and test set
trainingLabels2 = trainingset2.Labels;
developmentlabel2 = developmentset2.Labels;
testinglabel2 = testingsetdata2.Labels;
%% Resize and Augment images for the input of RESnET MODEL
 trainingset2.ReadFcn = @(filename)readAndPreprocessImage(filename);
inputSize = net.Layers(1).InputSize;
imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-20,20], ...
    'RandXTranslation',[-3 3], ...
    'RandYTranslation',[-3 3]);
augimdsTrain2 = augmentedImageDatastore(inputSize(1:2),trainingset2, ...
    'DataAugmentation',imageAugmenter);

%% Features extraction based on the last average pooling layer of ResNet for the training set 
trainingFeatures2 = activations(net,augimdsTrain2, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
%% Features extraction based on the last average pooling layer of ResNet for the development set 
 developmentset2.ReadFcn = @(filename)readAndPreprocessImage(filename);
% Extract development set features using the CNN
developmentFeatures2 = activations(net,developmentset2, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
%% Features extraction based on the last average pooling layer of ResNet for the testing set 
% Extract testing set features using the CNN
testingsetdata2.ReadFcn = @(filename)readAndPreprocessImage(filename);
testingFeatures2 = activations(net, testingsetdata2, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
 %%
% GRU
% Converting data into RNN FORMAT
rng(13);
trainf = {};
trainf{end+1} =  trainingFeatures2;

trainlabl = {};
trainlabl{end+1} = trainingLabels2';

train1 = {};
train1{end+1} = developmentFeatures2;
% 
train2 = {};
train2{end+1} = developmentlabel2';


numFeatures = 1920;
 numHiddenUnits =500;
numClasses = 2;
layers = [ ...
    sequenceInputLayer(numFeatures)
          gruLayer(numHiddenUnits,'OutputMode','sequence','RecurrentWeightsInitializer','he')
     fullyConnectedLayer(numClasses,'WeightsInitializer','he')
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
     'ExecutionEnvironment','gpu', ... 
       'InitialLearnRate',0.0001, ...
    'MaxEpochs',1500, ...
    'ValidationData',{train1,train2}, ...
    'ValidationFrequency',30, ...
     'SequenceLength','longest', ...
    'Plots','training-progress',...
   'OutputFcn',@(info)stopIfAccuracyNotImproving(info,3));

lstm2 = trainNetwork(trainf',trainlabl,layers,options);

[predictedLabels4, devlp_scores2] = classify(lstm2, developmentFeatures2);
% Converting labels into numerical form
 numericLabels1 = grp2idx(developmentlabel2);
 numericLabels1(numericLabels1==2)= -1;
 numericLabels1(numericLabels1==1)= 1;
 devlpscores2 =devlp_scores2';

  % Evaluation for testing set for EER 
 [TPR,TNR,Info]=vl_roc(numericLabels1,devlpscores2(:,1));
 EER = Info.eer*100
 threashold = Info.eerThreshold;

 % Evaluation for testing set for HTER 
 [predictedLabels2, test_scores2] = classify(lstm2, testingFeatures2);
 testscores2 = test_scores2';
 numericLabels = grp2idx(testinglabel2);
 numericLabels(numericLabels==2)= -1;
 numericLabels(numericLabels==1)= 1;
 real_scores1 =  testscores2(numericLabels==1);
 attack_scores2 =  testscores2(numericLabels==-1);
 FAR = sum(attack_scores2>threashold) / numel(attack_scores2)*100;
 FRR = sum(real_scores1<=threashold) / numel(real_scores1)*100;
 HTER2 = (FAR+FRR)/2
 
[x2,y2,threshold,AUC2] = perfcurve( numericLabels, testscores2(:,1),1);
AUC2
 plot(x2,y2,'-m','LineWidth',1.8,'MarkerSize',1.8)
 grid on
 hold on
 
  %% Input for Training set FOR DATASET 3
rootFolder = fullfile('D:\saliency based video summarzation\OMI\Train set');
categories  = {'real','attack'};
trainingset3 = imageDatastore(fullfile(rootFolder, categories), 'IncludeSubfolders',true, ...
 'LabelSource','foldernames');
%% Input for development set 
rootFolder = fullfile('D:\saliency based video summarzation\OMI\Test set');
categories  = {'real','attack'};
developmentset3 = imageDatastore(fullfile(rootFolder, categories), 'IncludeSubfolders',true, ...
 'LabelSource','foldernames');
%% Input for Testing set 
rootFolder = fullfile('D:\saliency based video summarzation\CASIA Dataset\testing set');
categories = {'real','attack'};
testingsetdata3 = imageDatastore(fullfile(rootFolder, categories),  'IncludeSubfolders',true, ...
 'LabelSource','foldernames');
%% Extracting  labels for training, development, and test set
trainingLabels3 = trainingset3.Labels;
developmentlabel3 = developmentset3.Labels;
testinglabel3 = testingsetdata3.Labels;
%% Resize and augment images for the input of RESnET MODEL
trainingset3.ReadFcn = @(filename)readAndPreprocessImage(filename);
inputSize = net.Layers(1).InputSize;
imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-20,20], ...
    'RandXTranslation',[-3 3], ...
    'RandYTranslation',[-3 3]);
augimdsTrain3 = augmentedImageDatastore(inputSize(1:2),trainingset3, ...
    'DataAugmentation',imageAugmenter);

%% Features extraction based on the last average pooling layer of ResNet for the training set 
trainingFeatures3 = activations(net, augimdsTrain3, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
%% Features extraction based on the last average pooling layer of ResNet for the development set 
 developmentset3.ReadFcn = @(filename)readAndPreprocessImage(filename);
% Extract development set features using the CNN
developmentFeatures3 = activations(net,developmentset3, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
%% Features extraction based on the last average pooling layer of ResNet for the testing set 
% Extract testing set features using the CNN
testingsetdata3.ReadFcn = @(filename)readAndPreprocessImage(filename);
testingFeatures3 = activations(net, testingsetdata3, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
 %%
% GRU 
% Converting data into RNN FORMAT
 rng(3);
trainf = {};
trainf{end+1} =  trainingFeatures3;

trainlabl = {};
trainlabl{end+1} = trainingLabels3';

train1 = {};
train1{end+1} = developmentFeatures3;
% 
train2 = {};
train2{end+1} = developmentlabel3';


numFeatures = 1920;
 numHiddenUnits =500;
numClasses = 2;
layers = [ ...
    sequenceInputLayer(numFeatures)
           gruLayer(numHiddenUnits,'OutputMode','sequence','RecurrentWeightsInitializer','he')
     fullyConnectedLayer(numClasses,'WeightsInitializer','he')
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
     'ExecutionEnvironment','gpu', ... 
       'InitialLearnRate',0.0001, ...
    'MaxEpochs',1500, ...
    'ValidationData',{train1,train2}, ...
    'ValidationFrequency',30, ...
     'SequenceLength','longest', ...
    'Plots','training-progress',...
   'OutputFcn',@(info)stopIfAccuracyNotImproving(info,3));

lstm3 = trainNetwork(trainf',trainlabl,layers,options);

[predictedLabels4, devlp_scores4] = classify(lstm3, developmentFeatures3);
% Converting labels into numerical form
 numericLabels1 = grp2idx(developmentlabel3);
 numericLabels1(numericLabels1==2)= -1;
 numericLabels1(numericLabels1==1)= 1;
 devlpscores4 =devlp_scores4';

  % Evaluation for testing set for EER
 [TPR,TNR,Info]=vl_roc(numericLabels1,devlpscores4(:,1));
 EER = Info.eer*100
 threashold = Info.eerThreshold;

 % Evaluation for testing set for HTER 
 [predictedLabels2, test_scores3] = classify(lstm3, testingFeatures3);
 testscores3 = test_scores3';
 numericLabels = grp2idx(testinglabel3);
 numericLabels(numericLabels==2)= -1;
 numericLabels(numericLabels==1)= 1;
 real_scores1 =  testscores3(numericLabels==1);
 attack_scores2 =  testscores3(numericLabels==-1);
 FAR = sum(attack_scores2>threashold) / numel(attack_scores2)*100;
 FRR = sum(real_scores1<=threashold) / numel(real_scores1)*100;
 HTER3 = (FAR+FRR)/2
[x3,y3,threshold,AUC3] = perfcurve( numericLabels, testscores3(:,1),1);
AUC3
 plot(x3,y3,'-b','LineWidth',1.8,'MarkerSize',1.8)
 grid on
 hold on
 
 %% Input for Training set FOR DATASET 4
rootFolder = fullfile('D:\saliency based video summarzation\OCI\Trainset');
categories  = {'real','attack'};
trainingset4 = imageDatastore(fullfile(rootFolder, categories), 'IncludeSubfolders',true, ...
 'LabelSource','foldernames');

%% Input for development set 
rootFolder = fullfile('D:\saliency based video summarzation\OCI\Testset');
categories  = {'real','attack'};
developmentset4 = imageDatastore(fullfile(rootFolder, categories), 'IncludeSubfolders',true, ...
 'LabelSource','foldernames');
%% Input for Testing set 
rootFolder = fullfile('D:\saliency based video summarzation\MSU dataset\testing set');
categories = {'real','attack'};
testingsetdata4 = imageDatastore(fullfile(rootFolder, categories),  'IncludeSubfolders',true, ...
 'LabelSource','foldernames');
%% Extracting  labels for training, development, and test set
trainingLabels4 = trainingset4.Labels;
developmentlabel4 = developmentset4.Labels;
testinglabel4 = testingsetdata4.Labels;
%% Input for Deep learning model
%% Resize images for the input of RESnET MODEL
trainingset4.ReadFcn = @(filename)readAndPreprocessImage(filename);
inputSize = net.Layers(1).InputSize;
imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-20,20], ...
    'RandXTranslation',[-3 3], ...
    'RandYTranslation',[-3 3]);
augimdsTrain4 = augmentedImageDatastore(inputSize(1:2),trainingset4, ...
    'DataAugmentation',imageAugmenter);

%% Features extraction based on the last average pooling layer of ResNet for the training set 
trainingFeatures4 = activations(net, augimdsTrain4, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
%% Features extraction based on the last average pooling layer of ResNet for the development set 
 developmentset4.ReadFcn = @(filename)readAndPreprocessImage(filename);
% Extract development set features using the CNN
developmentFeatures4 = activations(net,developmentset4, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
%% Features extraction based on the last average pooling layer of ResNet for the testing set 
% Extract testing set features using the CNN
testingsetdata4.ReadFcn = @(filename)readAndPreprocessImage(filename);
testingFeatures4 = activations(net, testingsetdata4, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
 %%
% GRU
% Converting data into RNN FORMAT
rng(13);
trainf = {};
trainf{end+1} =  trainingFeatures4;

trainlabl = {};
trainlabl{end+1} = trainingLabels4';

train1 = {};
train1{end+1} = developmentFeatures4;
% 
train2 = {};
train2{end+1} = developmentlabel4';


numFeatures = 1920;
 numHiddenUnits =500;
numClasses = 2;
layers = [ ...
    sequenceInputLayer(numFeatures)
       gruLayer(numHiddenUnits,'OutputMode','sequence','RecurrentWeightsInitializer','he')
     fullyConnectedLayer(numClasses,'WeightsInitializer','he')
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
     'ExecutionEnvironment','gpu', ... 
       'InitialLearnRate',0.0001, ...
    'MaxEpochs',1500, ...
    'ValidationData',{train1,train2}, ...
    'ValidationFrequency',30, ...
     'SequenceLength','longest', ...
    'Plots','training-progress',...
   'OutputFcn',@(info)stopIfAccuracyNotImproving(info,3));

lstm4 = trainNetwork(trainf',trainlabl,layers,options);

[predictedLabels4, devlp_scores5] = classify(lstm4, developmentFeatures4);
% Converting labels into numerical form
 numericLabels1 = grp2idx(developmentlabel4);
 numericLabels1(numericLabels1==2)= -1;
 numericLabels1(numericLabels1==1)= 1;
 devlpscores5 =devlp_scores5';

 % Evaluation for testing set for EER
 [TPR,TNR,Info]=vl_roc(numericLabels1,devlpscores5(:,1));
 EER = Info.eer*100
 threashold = Info.eerThreshold;

 % Evaluation for testing set for HTER 
 [predictedLabels2, test_scores4] = classify(lstm4, testingFeatures4);
 testscores4 = test_scores4';
 numericLabels = grp2idx(testinglabel4);
 numericLabels(numericLabels==2)= -1;
 numericLabels(numericLabels==1)= 1;
 real_scores1 =  testscores4(numericLabels==1);
 attack_scores2 =  testscores4(numericLabels==-1);
 FAR = sum(attack_scores2>threashold) / numel(attack_scores2)*100;
 FRR = sum(real_scores1<=threashold) / numel(real_scores1)*100;
 HTER4 = (FAR+FRR)/2
[x4,y4,threshold,AUC4] = perfcurve(numericLabels, testscores4(:,1),1);
AUC4
plot(x4,y4,'-c','LineWidth',1.8,'MarkerSize',1.8)
grid on
hold off

legend('I&C&M to O', 'O&C&M to I','O&M&I to C', 'O&C&I to M')


  %%
 % Resize images 
     function Iout = readAndPreprocessImage(filename)

       I = imread(filename);

         if ismatrix(I)
            I = cat(3,I,I,I);
         end
     

           Iout = imresize(I, [224 224]);
            
     end
    %%
 % Function for early stopping
 
 function stop = stopIfAccuracyNotImproving(info,N)

stop = false;

% Keep track of the best validation accuracy and the number of validations for which
% there has not been an improvement of the accuracy.
persistent bestValAccuracy
persistent valLag

% Clear the variables when training starts.
if info.State == "start"
    bestValAccuracy = 0;
    valLag = 0;
    
elseif ~isempty(info.ValidationLoss)
    
    % Compare the current validation accuracy to the best accuracy so far,
    % and either set the best accuracy to the current accuracy, or increase
    % the number of validations for which there has not been an improvement.
    if info.ValidationAccuracy > bestValAccuracy
        valLag = 0;
        bestValAccuracy = info.ValidationAccuracy;
    else
        valLag = valLag + 1;
    end
    
    % If the validation lag is at least N, that is, the validation accuracy
    % has not improved for at least N validations, then return true and
    % stop training.
    if valLag >= N
        stop = true;
    end
    
end

end