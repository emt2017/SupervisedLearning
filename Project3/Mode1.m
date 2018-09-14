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
[evec,eval]=eig(v);

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

LDADistance=pdist2(y',tFinal','Euclidean');

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

%////////////////////////////////end of LDA//////////////////////////////


%face data is 92x112

%This code to apply PCA (Principal Component Analysis) 

% Remember that each column of the data matrix(input matrix) represent one image or pattern  
% Note: the data here represent two classes
% Class 1: data(:,1:4)
% Class 2: data(:,5:8)

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


[r,c] = size(data);
% Compute the mean of the data matrix "The mean of each row"
m = mean(data')';
% Subtract the mean from each image [Centering the data]
d=data-repmat(m,1,c);


% Compute the covariance matrix (co)
co=d*d';

% Compute the eigen values and eigen vectors of the covariance matrix
[eigvector,eigvl]=eig(co);


% Sort the eigen vectors according to the eigen values
eigvalue = diag(eigvl);
[junk, index] = sort(eigvalue,'descend');
eigvalue = eigvalue(index);
eigvector = eigvector(:, index);

% Compute the number of eigen values that greater than zero (you can select any threshold)
count1=0;
for i=1:size(eigvalue,1)
    if(eigvalue(i)>0)
        count1=count1+1;
    end
end

% We can use all the eigen vectors but this method will increase the
% computation time and complixity
%vec=eigvector(:,:);

% And also we can use the eigen vectors that the corresponding eigen values is greater than zero(Threshold) and this method will decrease the
% computation time and complixity

vec=eigvector(:,1:count1);

% Compute the feature matrix (the space that will use it to project the testing image on it)
x=vec'*d;

% If you have test data do the following
%send one picture through at a time to test against the data?
%111110000000000
%000001111100000
%000000000011111
%t=[1;1]  % this test data is close to the first class
%Subtract the mean from the test data
t=t-m;
%Project the testing data on the space of the training data
t=vec'*t;

%calculate euclidean distance between train and test sets
PCADistance = pdist2(x', t', 'Euclidean');

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

% %//////////////Normalize pca distances and LDA distances////////////////
% normalizedPCADist = normc(PCADistance);
% normalizedLDADist = normc(LDADistance);
% 
% %//////////////Test normalized//////////////////////////////////
% resultMax = max(normalizedPCADist,normalizedLDADist);
% resultMin = min(normalizedPCADist,normalizedLDADist);
% resultAvg = (normalizedPCADist + normalizedLDADist)/2;
% 
% %create ezroc graph
% [PCAROC] = ezroc3(normalizedPCADist,targets,2,'',1)
% [LDAROC] = ezroc3(normalizedLDADist,targets,2,'',1)
% [AvgROC] = ezroc3(resultAvg,targets,2,'',1)
% [MaxROC] = ezroc3(resultMax,targets,2,'',1)
% [MinROC] = ezroc3(resultMin,targets,2,'',1)
% 
% plot(AvgROC(2,:),AvgROC(1,:),MaxROC(2,:),MaxROC(1,:),MinROC(2,:),MinROC(1,:),LDAROC(2,:),LDAROC(1,:),PCAROC(2,:),PCAROC(1,:))
% xlabel('FNR')
% ylabel('TPR')
% legend('Average','Max','Min','LDA','PCA')
%/////////////End Test normalized///////////////////////////////


resultMax = max(PCADistance,LDADistance);
resultMin = min(PCADistance,LDADistance);
resultAvg = (PCADistance + LDADistance)/2;

%////////////////////////////////Mode 1//////////////////////////////////
%create ezroc graph
[PCAROC] = ezroc3(PCADistance,targets,2,'',1)
[LDAROC] = ezroc3(LDADistance,targets,2,'',1)
[AvgROC] = ezroc3(resultAvg,targets,2,'',1)
[MaxROC] = ezroc3(resultMax,targets,2,'',1)
[MinROC] = ezroc3(resultMin,targets,2,'',1)

plot(AvgROC(2,:),AvgROC(1,:),MaxROC(2,:),MaxROC(1,:),MinROC(2,:),MinROC(1,:),LDAROC(2,:),LDAROC(1,:),PCAROC(2,:),PCAROC(1,:))
xlabel('FNR')
ylabel('TPR')
legend('Average','Max','Min','LDA','PCA')

%////////////////////////////////Mode 2//////////////////////////////////
%create Mode 2 targets 200x40
targetsM2 = [zeros(5,1) ones(5,39)];
tempTargetM2 = [];
    k = 1;
    j = 38;
    
for i = 3:40
    tempTargetM2 = [ones(5,k) zeros(5,1) ones(5,j)];
    targetsM2 = cat(1, targetsM2, tempTargetM2);
    
    k = k + 1;
    j = j - 1;
end

tempTargetM2 = [ones(5,39) zeros(5,1)];

targetsM2 = cat(1, targetsM2, tempTargetM2);

%average distances PCA
PCADistance;
PCADistanceAvg = [];
tempDistAvg = [];

for i = 1:200
    for j = 1:40
        tempDistAvg = cat(2,tempDistAvg,mean(PCADistance(i,5*j-4:5*j)));
    end
    PCADistanceAvg = cat(1,PCADistanceAvg,tempDistAvg);
    tempDistAvg = [];
end


%average distances PCA
LDADistance;
LDADistanceAvg = [];
tempDistAvg = [];

for i = 1:200
    for j = 1:40
        tempDistAvg = cat(2,tempDistAvg,mean(LDADistance(i,5*j-4:5*j)));
    end
    LDADistanceAvg = cat(1,LDADistanceAvg,tempDistAvg);
    tempDistAvg = [];
end



%plot LDA and PCA
%create ezroc graph
[PCAROC] = ezroc3(PCADistance,targets,2,'',1)
[LDAROC] = ezroc3(LDADistance,targets,2,'',1)
[PCAROCAvg]=ezroc3(PCADistanceAvg,targetsM2,2,'',1)
[LDAROCAvg]=ezroc3(LDADistanceAvg,targetsM2,2,'',1)

fig1 = figure
plot(LDAROCAvg(2,:),LDAROCAvg(1,:),LDAROC(2,:),LDAROC(1,:))
xlabel('FNR')
ylabel('TPR')
legend('LDA Avg','LDA')

fig2 = figure
plot(PCAROCAvg(2,:),PCAROCAvg(1,:),PCAROC(2,:),PCAROC(1,:))
xlabel('FNR')
ylabel('TPR')
legend('PCA Avg','PCA')




