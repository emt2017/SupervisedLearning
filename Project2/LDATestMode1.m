clc
clear all
close all

%This code to apply LDA (Linear Discriminant Analysis) 


% This example deals with 2 classes
data = [1 2 1 0;2 3 2 1;3 3 3 1;4 5 3 2;5 5 5 3]  % the first class 5 observations
% the second class 6 observations
% scatter(c1(:,1),c1(:,2),6,'r'),hold on;
% scatter(c2(:,1),c2(:,2),6,'b');

numClasses = 2;
numPics = 5;
picSize = 2;

% Number of observations of each class
numObservations = size(data,1);

mu = 0;
%Mean of all classes
for i=1:2
    if(i==1)
        mu = mu + mean(data(:, i:picSize)) %mu1
    else
        mu = mu + mean(data(:,(i-1)*picSize+1:i*picSize))%mu2,3,4,5...
    end
end

% Average of the mean of all classes
mu = mu/numClasses

sw = 0;
for i=1:2
    d = 0;
    s = 0;
    % Center the data (data-mean)
    if(i==1)
        d = data(:, i:picSize)-repmat(mean(data(:, i:picSize)),numObservations,1) %c1
    else
        d = data(:,(i-1)*picSize+1:i*picSize)-repmat(mean(data(:,(i-1)*picSize+1:i*picSize)),numObservations,1)%c2,3,4,5...
    end
    % Calculate the within class variance (SW)
    s=d'*d
    sw = sw + s
end
invsw=inv(sw)
%/////////////////////////////////////////////////////////////////////
% in case of two classes only use v
% v=invsw*(mu1-mu2)'

% if more than 2 classes calculate between class variance (SB)
SB = 0;
for i=1:2
    sb = 0;
    % Center the data (data-mean)
    if(i==1)
        sb = numObservations*(mean(data(:, i:picSize))-mu)'*(mean(data(:, i:picSize))-mu)
    else
        sb = numObservations*(mean(data(:,(i-1)*picSize+1:i*picSize))-mu)'*(mean(data(:,(i-1)*picSize+1:i*picSize))-mu)
    end
    % Calculate the within class variance (SW)
    SB = SB + sb
end
v=invsw*SB


% find eigne values and eigen vectors of the (v)
[evec,eval]=eig(v)

% Sort eigen vectors according to eigen values (descending order) and
% neglect eigen vectors according to small eigen values
% v=evec(greater eigen value)
% or use all the eigen vectors

% project the data of the first and second class respectively
%y2=data*v
%y1=c1*v

% project the data of all the classes
y = [];
for i=1:2
    yTemp = [];
    yTemp = data(:,(i-1)*picSize+1:i*picSize)*v;
    y = cat(2,y,yTemp);
end

t = data - mean(data')';
tFinal = [];
for i=1:2
    tTemp = [];
    tTemp = t(:,(i-1)*picSize+1:i*picSize)*v;
    tFinal = cat(2,tFinal,tTemp);
end
tFinal

D=pdist2(y',tFinal','Euclidean');

targets = [0 0 1 1;0 0 1 1;1 1 0 0;1 1 0 0]

%create ezroc graph
ezroc3(D,targets,2,'',1)

% %sort eigen values and eigen vectors of the covraiancce matrix
% eigenVal = diag(eval);
% [junk, index] = sort(eigenVal,'descend');
% eigenVal = eigenVal(index);
% evec = evec(:,index)
% 
% %Compute the number of eigen values that greater than zero (you can
% %select any threshold)
% 
% count1 = 0;
% for i=1:size(eigenVal,1)
%     if(eigenVal(i)>0)
%         count1=count1+1;
%     end
% end
% vec = evec(:,1:count1);
% 
% [r,c]=size(data);
% m= mean(data')'
% 
% rep = repmat(m,1,c);
% d=data-repmat(m,1,c);
% 
% x = d*vec'
% 


