clc
clear all
close all
%face data is 92x112

%This code to apply PCA (Principal Component Analysis) 

% Remember that each column of the data matrix(input matrix) represent one image or pattern  
% Note: the data here represent two classes
% Class 1: data(:,1:4)
% Class 2: data(:,5:8)

load FaceCellArrayData.mat

% Get the number of pixels in each image
imageWidth = 92;
imageHeight = 112;
inputSize = imageWidth*imageHeight;

% Turn the train images into vectors and put them in a matrix
data=FacePCACellArrayDataP2;
data = zeros(inputSize,numel(FacePCACellArrayDataP2));
for i = 1:numel(FacePCACellArrayDataP2)
    data(:,i) = FacePCACellArrayDataP2{i}(:);
end

% Turn the train images into vectors and put them in a matrix
%train matrix last 15 subject 1:5 images
x=FaceTrainCellArrayDataP2;
x = zeros(inputSize,numel(FaceTrainCellArrayDataP2));
for i = 1:numel(FaceTrainCellArrayDataP2)
    x(:,i) = FaceTrainCellArrayDataP2{i}(:);
end

% Turn the test images into vectors and put them in a matrix
%test matrix last 15 subjects 6:10 images
t=FaceTestCellArrayDataP2;
t = zeros(inputSize,numel(FaceTestCellArrayDataP2));
for i = 1:numel(FaceTestCellArrayDataP2)
    t(:,i) = FaceTestCellArrayDataP2{i}(:);
end


%t = t(:,1:10);

%Create eigen PCA with first 25 subject picture 1:10
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
%train the data
x=x-m;
x=vec'*x;

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
D = pdist2(x', t', 'Euclidean');


%create targets
targets = [zeros(5,5) ones(5,70)];
tempTarget = [];
    k = 5;
    j = 65;
    
for i = 28:40
    tempTarget = [ones(5,k) zeros(5,5) ones(5,j)];
    targets = cat(1, targets, tempTarget);
    
    k = k + 5;
    j = j - 5;
end

tempTarget = [ones(5,70) zeros(5,5)];

targets = cat(1, targets, tempTarget);

%create ezroc graph
ezroc3(D,targets,2,'',1)
