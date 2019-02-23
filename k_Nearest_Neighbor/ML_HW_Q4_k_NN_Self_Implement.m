%Code for Problem 4
%Using self-writing K-NN calssifier: KNN

clc;
clear;
%% set number of training data and test data
n_train=2000;
n_test=1000;

%% 1-NN classifier
k=1;
err1nn=zeros(11,1);
for p=1:10:101
    
%generate training data
[x_train,y_train]=generate_data(p,n_train);

%generate test data
[x_test,y_test]=generate_data(p,n_test);

%get the prediction for the test_data with model trained with trainning
%data
y_predict_1 = KNN(1,x_train,y_train,x_test);

%error rate

err1nn(k)=sum((y_test-y_predict_1')~=0)/n_test;
k=k+1;
end

%% 3-NN classifier
m=1;
err3nn=zeros(11,1);
for p=1:10:101
    
%generate training data
[x_train,y_train]=generate_data(p,n_train);

%generate test data
[x_test,y_test]=generate_data(p,n_test);

%get prediction for the test_data with model trained with trainning data
y_predict_3 = KNN(3,x_train,y_train,x_test);

%error rate
err3nn(m)=sum((y_test-y_predict_3')~=0)/n_test;
m=m+1;
end
%% Plot the error rate as a function of dimension p
figure
p0=1:10:101;
plot(p0,err1nn,'-*','MarkerSize',7)
hold on
plot(p0,err3nn,'--o','MarkerSize',7)
xlabel('dimension');
ylabel('error_rate');
legend('1-NN','3-NN','Location','Best')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Functions used above
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Data generating function %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ x,y ] = generate_data(p,n)
%This function is used for generate 'n' points data from 
%two equally two equally weighted spherical Gaussians
% N(0,0..0,I),N(3,0...0),I) in 'p' dimension;

%set up mean and variance for spherical Gaussians
mean1=zeros(p,1);
mean2=mean1;
mean2(1,1)=3;
cov=eye(p);

%generate data from two spherical Gaussians
sample1=mvnrnd(mean1,cov,n);
sample2=mvnrnd(mean2,cov,n);
x=zeros(n,p);

%Using a random number generator (between 0 and 1) to choose data for 
%trainning sample and test sample
%Assign labels to different sample according to the random number:
%if random number <0.5,y=1; else,y=-1
for i=1:n
    r=rand;
    
    if r<0.5
        x(i,:)=sample1(i,:);
        y(i)=1;
    else
        x(i,:)=sample2(i,:);
        y(i)=-1;
    end
    
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%K-Nearest Neighbor classifier%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function predicted_labels = KNN(k,data,labels,t_data)
%KNN: classifying using k-nearest neighbors algorithm. The nearest neighbors
%search method is euclidean distance
%Usage:
%       predicted_labels = KNN_(3,training,training_labels,testing)
%Input:
%       - k: number of nearest neighbors
%       - data: (NxD) training data; 
%       - labels: training labels 
%       - t_data: (MxD) testing data; 
%Output:
%       - predicted_labels: the predicted labels based on the k-NN algorithm
%       
    
%initialization
predicted_labels=zeros(size(t_data,1),1);
ed=zeros(size(t_data,1),size(data,1)); %ed: (MxN) euclidean distances 
ind=zeros(size(t_data,1),size(data,1)); %corresponding indices (MxN)


%calc euclidean distances between each testing data point and the training data samples
for test_point=1:size(t_data,1)
    for train_point=1:size(data,1)
%calc and store sorted euclidean distances with corresponding indices
        ed(test_point,train_point)=sqrt(...
            sum((t_data(test_point,:)-data(train_point,:)).^2));
    end
    [ed(test_point,:),ind(test_point,:)]=sort(ed(test_point,:));
end

%find the nearest k for each data point of the testing data
k_nn=ind(:,1:k);

%get the majority vote 
for i=1:size(k_nn,1)
    options=unique(labels(k_nn(i,:)'));
    max_count=0;
    max_label=0;
    for j=1:length(options)
        L=length(find(labels(k_nn(i,:)')==options(j)));
        if L>max_count
            max_label=options(j);
            max_count=L;
        end
    end
    predicted_labels(i)=max_label;
end