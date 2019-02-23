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

