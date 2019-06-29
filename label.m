% 处理类标签给SVM用的

%训练数据的标签
a=1;
for i=1:100
    H_train1(1,i)=a; 
    if rem(i,10)==0
        a = a+1;
    end
end
H_train=H_train1(1,:)';

%测试数据的标签
b=1;
for i=1:200
    H_test1(1,i)=b; 
    if rem(i,20)==0
        b = b+1;
    end
end
H_test=H_test1(1,:)';

