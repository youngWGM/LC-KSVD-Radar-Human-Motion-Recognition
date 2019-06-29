% ========================================================================
% 分类 
% [prediction, accuracy, err] = classification(D, W, data, Hlabel,
%                                       sparsity)
% 输入
%       D               -学习到的字典
%       W               -learned classifier parameters学习到的分类参数
%       data            -testing features 测试数据
%       Hlabel          -labels matrix for testing feature 测试数据的真实类别
%       iterations      -iterations for KSVD 迭代次数
%       sparsity        -sparsity threshold 稀疏阈值
% outputs
%       prediction      -predicted labels for testing features 预测的类别
%       accuracy        -classification accuracy 准确率
%       err             -misclassfication information  误分类信息
%                       [errid featureid groundtruth-label predicted-label]
% ========================================================================

function [prediction, accuracy, err] = classification(D, W, data, Hlabel, sparsity)

% 稀疏编码
G = D'*D;
Gamma = omp(D'*data,G,sparsity);

% classify process
errnum = 0;
err = [];
prediction = [];
for featureid=1:size(data,2)
    spcode = Gamma(:,featureid);
    score_est =  W * spcode;
    score_gt = Hlabel(:,featureid);
    [maxv_est, maxind_est] = max(score_est);  % classifying
    [maxv_gt, maxind_gt] = max(score_gt);
    prediction = [prediction maxind_est];
    if(maxind_est~=maxind_gt)
        errnum = errnum + 1;
        err = [err;errnum featureid maxind_gt maxind_est];
    end
end
accuracy = (size(data,2)-errnum)/size(data,2);



