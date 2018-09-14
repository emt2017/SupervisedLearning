clc
clear all
close all

%face data is 92x112

%This code to apply PCA (Principal Component Analysis) 

% Remember that each column of the data matrix(input matrix) represent one image or pattern  
% Note: the data here represent two classes
% Class 1: data(:,1:4)
% Class 2: data(:,5:8)

%load FaceCellArrayData.mat
load('\\kc.umkc.edu\kc-users\home\e\emt9q7\My Documents\MATLAB\SupervisedLearning\Project1\FaceCellArrayData.mat');

data=FaceTrainCellArrayData;
%my data is not oranized correctly make sure to go back to organize your
%data
%correctly!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

% Get the number of pixels in each image
imageWidth = 92;
imageHeight = 112;
inputSize = imageWidth*imageHeight;

% Turn the train images into vectors and put them in a matrix
data = zeros(inputSize,numel(FaceTrainCellArrayData));
for i = 1:numel(FaceTrainCellArrayData)
    data(:,i) = FaceTrainCellArrayData{i}(:);
end

% Turn the test images into vectors and put them in a matrix
t=FaceTestCellArrayData;
t = zeros(inputSize,numel(FaceTestCellArrayData));
for i = 1:numel(FaceTestCellArrayData)
    t(:,i) = FaceTestCellArrayData{i}(:);
end

%t = t(:,1:10);
% the second class 6 observations
% scatter(c1(:,1),c1(:,2),6,'r'),hold on;
% scatter(c2(:,1),c2(:,2),6,'b');

numClasses = 40;
numPics = 200;
picSize = 5;

% Number of observations of each class
numObservations = size(data,1);

mu = 0;
%Mean of all classes
for i=1:40
    if(i==1)
        mu = mu + mean(data(:, i:picSize)) %mu1
    else
        mu = mu + mean(data(:,(i-1)*picSize+1:i*picSize))%mu2,3,4,5...
    end
end

% Average of the mean of all classes
mu = mu/numClasses

sw = 0;
for i=1:40
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
for i=1:40
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
for i=1:40
    yTemp = [];
    yTemp = data(:,(i-1)*picSize+1:i*picSize)*v;
    y = cat(2,y,yTemp);
end

tFinal = [];
for i=1:40
    tTemp = [];
    tTemp = t(:,(i-1)*picSize+1:i*picSize)*v;
    tFinal = cat(2,tFinal,tTemp);
end
tFinal

D=pdist2(y',tFinal','Euclidean');

%create targets
targets = [zeros(5,5) ones(5,195)];
tempTarget = [];
    k = 5;
    j = 190;
    
for i = 3:40
    tempTarget = [ones(5,k) zeros(5,5) ones(5,j)];
    targets = cat(1, targets, tempTarget);
    
    k = k + 5;
    j = j - 5;
end


tempTarget = [ones(5,195) zeros(5,5)];

targets = cat(1, targets, tempTarget);

%create ezroc graph
ezroc3(D,targets,2,'',1)


