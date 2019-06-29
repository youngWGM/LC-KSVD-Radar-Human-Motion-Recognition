%画混淆矩阵

A=prediction2;
num=[20 20 20 20 20 20 20 20 20 20];%10类，每类20组
name=cell(1,10);
name{1}='1';name{2}='2';name{3}='3';name{4}='4';name{5}='5';
name{6}='6';name{7}='7';name{8}='8';name{9}='9';name{10}='10';
compute_confusion_matrix(A,num,name)
