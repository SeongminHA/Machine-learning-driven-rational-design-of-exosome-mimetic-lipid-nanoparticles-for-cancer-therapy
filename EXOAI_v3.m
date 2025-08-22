% hybrid_exosome_prediction.m (Enhanced with real-time and final plotting)
% This script reads exosome composition data from Excel or CSV (rows 2 to 10000),
% applies preprocessing, splits A–E composition ratios 70/30 for train/test,
% trains separate stacking ensembles to predict performance metrics (F–H),
% and recommends the top 20 composition conditions with detailed plots.

%% 1. Prompt user to select a data file (Excel or CSV)
[filename, filepath] = uigetfile({'*.xlsx;*.csv','Data Files (*.xlsx, *.csv)'}, 'Select the Data File');
if isequal(filename, 0)
    disp('File selection canceled.'); return;
end
fullpath = fullfile(filepath, filename);

%% 2. Read table (preserve names) and select rows 2–10000
opts = detectImportOptions(fullpath, 'VariableNamingRule','preserve');
dataFull = readtable(fullpath, opts);
if height(dataFull) < 10000
    error('The data must contain at least 10000 rows.');
end
% Use numeric data from row 2 to 10000
data = dataFull{2:10000, :}; % raw numeric array

%% 3. Extract columns A–E (composition ratios) and normalize to sum=100
ratios = data(:,1:5);
rowSum = sum(ratios, 2);
cf = 100 ./ rowSum; cf(rowSum==0)=1;
ratios = round(ratios .* cf);

%% 4. Split composition rows 70% train, 30% test randomly
n = size(ratios,1);
perm = randperm(n);
idxTrain = perm(1:round(0.7*n));
idxTest  = perm(round(0.7*n)+1:end);
ratios_train = ratios(idxTrain,:);
ratios_test  = ratios(idxTest ,:);

%% 5. Extract performance metrics F–H as targets
targets = data(:,6:8);
tgt_train = targets(idxTrain,:);
tgt_test  = targets(idxTest ,:);

%% 6. Standardize predictors and targets on training set
muR = mean(ratios_train); sigmaR = std(ratios_train);
r_train = (ratios_train - muR) ./ sigmaR;
r_test  = (ratios_test  - muR) ./ sigmaR;

yt_mu = mean(tgt_train); yt_std = std(tgt_train);
y_train = (tgt_train - yt_mu) ./ yt_std;
y_test  = (tgt_test  - yt_mu) ./ yt_std;

%% 7. Define and train base learners with real-time plot for NN
metaModels = cell(3,1);
finalNNs   = cell(3,1);
finalGBs   = cell(3,1);
finalSVRs  = cell(3,1);
trainingInfo = cell(3,1);
for t = 1:3
    ytr = y_train(:,t);
    nTrain = numel(ytr);
    % 5-fold regression CV
    K = 5;
    cvp = cvpartition(nTrain, 'KFold', K);
    oofNN  = zeros(nTrain,1);
    oofGB  = zeros(nTrain,1);
    oofSVR = zeros(nTrain,1);
    for kf = 1:K
        trIdx = training(cvp, kf);
        vaIdx = test(cvp, kf);
        % Neural Network with real-time training plot
        layers = [featureInputLayer(5,'Name','input')
                  fullyConnectedLayer(10,'Name','fc1')
                  reluLayer('Name','relu')
                  fullyConnectedLayer(1,'Name','fc2')
                  regressionLayer('Name','output')];
        opts = trainingOptions('sgdm', ...
            'MaxEpochs',1000, 'MiniBatchSize',32, 'InitialLearnRate',0.01, ...
            'Plots','training-progress','Verbose',false);
        [net, info] = trainNetwork(r_train(trIdx,:), ytr(trIdx), layers, opts);
        oofNN(vaIdx) = predict(net, r_train(vaIdx,:));
        trainingInfo{t} = info;  % store last fold info for plotting
        % LSBoost
        mdlGB = fitrensemble(r_train(trIdx,:), ytr(trIdx), ...
            'Method','LSBoost','Learners',templateTree('MaxNumSplits',10), 'NumLearningCycles',100);
        oofGB(vaIdx) = predict(mdlGB, r_train(vaIdx,:));
        % SVR
        mdlSVR = fitrsvm(r_train(trIdx,:), ytr(trIdx), ...
            'KernelFunction','rbf','Standardize',true,'Verbose',0);
        oofSVR(vaIdx) = predict(mdlSVR, r_train(vaIdx,:));
    end
    % Meta-learner
    metaX = [oofNN, oofGB, oofSVR];
    metaModels{t} = fitrlinear(metaX, ytr, 'Learner','leastsquares');
    % Retrain base on full training set
    finalNNs{t}  = trainNetwork(r_train, ytr, layers, opts);
    finalGBs{t}  = fitrensemble(r_train, ytr, ...
        'Method','LSBoost','Learners',templateTree('MaxNumSplits',10),'NumLearningCycles',100);
    finalSVRs{t} = fitrsvm(r_train, ytr, 'KernelFunction','rbf','Standardize',true,'Verbose',0);
end

%% 8. Predict on test set and assemble performances
nTest = size(r_test,1);
preds = zeros(nTest,3);
for t = 1:3
    pNN  = predict(finalNNs{t}, r_test);
    pGB  = predict(finalGBs{t}, r_test);
    pSVR = predict(finalSVRs{t}, r_test);
    preds(:,t) = predict(metaModels{t}, [pNN, pGB, pSVR]);
end
% Rescale back to original scale
preds = preds .* yt_std + yt_mu;

%% 9. Evaluate metrics and plot final results
mseVals = mean((preds - tgt_test).^2);
r2Vals  = 1 - mseVals ./ var(tgt_test);
figure('Name','Performance Summary');
bar(categorical({'Target F','Target G','Target H'}), [mseVals; r2Vals]');
ylabel('Metric Value');
legend('MSE','R^2','Location','Best');
title('Final Regression Metrics per Target');

%% 10. Recommend top 20 compositions by average predicted score
scores = mean(preds,2);
[~,order] = sort(scores, 'descend');
top20 = ratios_test(order(1:20), :);

figure('Name','Top 20 Composition Ratios');
uicontrol('Style','text','Position',[20 400 300 200], ...
    'String', sprintf('Top 20 A–E ratios:\n%s', mat2str(top20)));

%% Optional: Plot training loss for each target NN
for t = 1:3
    info = trainingInfo{t};
    figure('Name',sprintf('NN Training Loss for Target %d',t));
    plot(info.TrainingLoss); xlabel('Iteration'); ylabel('Loss');
    title(sprintf('Neural Network Training Loss (Target %d)',t));
end
