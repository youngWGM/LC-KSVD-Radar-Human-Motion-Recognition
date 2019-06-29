%clear;
clc;
extractfeature1
addpath(genpath('.\ksvdbox'));  % add K-SVD box  导入KSVD工具箱
addpath(genpath('.\OMPbox')); % add sparse coding algorithem OMP  导入OMP工具箱
%load('trainingdata\featurevectors.mat','training_feats', 'testing_feats', 'H_train', 'H_test');


%% constant设置参数
sparsitythres = 5; % sparsity prior   这是什么？？？
sqrt_alpha = 4; % weights for label constraint term 标签约束项的权重
sqrt_beta = 2; % weights for classification err term 分类误差项的权重
dictsize = 100; % dictionary size 字典大小
iterations = 50; % iteration number 迭代50次
iterations4ini = 20; % iteration number for initialization 初始化的迭代次数20


%% dictionary learning process
% get initial dictionary Dinit and Winit
fprintf('\nLC-KSVD 初始化... ');
[Dinit,Tinit,Winit,Q_train] = initialization4LCKSVD(train_fea,H_train1,dictsize,iterations4ini,sparsitythres);%初始化字典
fprintf('完成!');


%run LC K-SVD Training (reconstruction err + class penalty)
% fprintf('\nDictionary learning by LC-KSVD1...');
% [D1,X1,T1,W1] = labelconsistentksvd1(train_fea,Dinit,Q_train,Tinit,H_train1,iterations,sparsitythres,sqrt_alpha);
% save('trainingdata\dictionarydata1.mat','D1','X1','W1','T1');
% fprintf('完成!');


% run LC k-svd training (reconstruction err + class penalty + classifier err)
fprintf('\nDictionary and classifier learning by LC-KSVD2...')
[D2,X2,T2,W2] = labelconsistentksvd2(train_fea,Dinit,Q_train,Tinit,H_train1,Winit,iterations,sparsitythres,sqrt_alpha,sqrt_beta);
save('trainingdata\dictionarydata2.mat','D2','X2','W2','T2');
fprintf('完成!');



% %% classification process
% [prediction1,accuracy1] = classification(D1, W1, test_fea, H_test1, sparsitythres);
% fprintf('\nLC-KSVD1 的识别准确率: %.03f ', accuracy1);


[prediction2,accuracy2] = classification(D2, W2, test_fea, H_test1, sparsitythres);
fprintf('\nLC-KSVD2 的识别准确率: %.03f ', accuracy2);


