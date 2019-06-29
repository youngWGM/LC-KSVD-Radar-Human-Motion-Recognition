
%数据预处理
%提取特征

clear
dims = 100; % dimension of random-face feature descriptor
IMG_H = 128; % 输入图片的大小
IMG_W = 128;

% 生成 random 矩阵
randmatrix = randn(dims,IMG_H*IMG_W);
l2norms = sqrt(sum(randmatrix.*randmatrix,2)+eps);
randmatrix = randmatrix./repmat(l2norms,1,size(randmatrix,2));


% Note that you should use the same randmatrix 
% when you iterate all the input images and extract descriptors
total_path = 'F:\matlab my file\Dictionary-learning-master\p2\';
jpg = '.jpg';
train_fea = [];
test_fea = [];
fea=[];
H_train1 = zeros(10);
H_test1 = zeros(10);
class = ['A','B','C','D','E','F','G','H','I','J'];
x = 1;
y = 1;
aa = 1;
for j = 1:10
for i = 1:30
    single_path = strcat(total_path,class(j));
    single_path = strcat(single_path,'\');
    single_path = strcat(single_path,num2str(i));
    single_path = strcat(single_path,jpg);
    img = imread(single_path);
    feature = double(img(:));
    all_img(:,aa)=feature(:);
    aa = aa + 1;
    randomfacefeature = randmatrix*feature;
    %设置标签
%     if i <= 20&&i>10
    if i>20
        H_train1(j, x) = 1;
        x = x+1;
    end
%     if i > 20||i<=10
    if i<=20
        H_test1(j, y) = 1;
        y = y+1;
    end
    fea = [fea, randomfacefeature];
end
end


all_img1 = all_img';    %所有图片矩阵转置300*16384
[coef,score,latent,t2] = pca(all_img1);  %pca处理
%YYY=all_img1*coef(:,10:100);   %根据pca进行降维
YYY = score(:,1:60);
YYY=YYY';  %转置
fea=[fea;YYY];



for k=1:300
%     if rem(k,30)>9&&rem(k,30)<=19
    if rem(k,30)>19
        train_fea = [train_fea,fea(:,k)];
%     elseif rem(k,30)<=9||rem(k,30)>19
    elseif rem(k,30)<=19
        test_fea = [test_fea,fea(:,k)];
    end
end




    