clc
clear all
close all

%Train
i = 1; %person 1-40
j = 1; %img 1-5
count = 0;
FaceTrainCellArrayData = cell(1, count);

for i = 1:40    
    for j = 1:5
        count = count + 1;  
        img = imread(['\\kc.umkc.edu\kc-users\home\e\emt9q7\My Documents\MATLAB\SupervisedLearning\Project1\Face_Data\s' num2str(i) '\' num2str(j) '.pgm']);
        FaceTrainCellArrayData{count}=img;
    end
end

%Test 
i = 1; %person 1-40
j = 1; %img 6-10
count = 0;
FaceTestCellArrayData = cell(1, count);

for i = 1:40    
    for j = 6:10
        count = count + 1;  
        img = imread(['\\kc.umkc.edu\kc-users\home\e\emt9q7\My Documents\MATLAB\SupervisedLearning\Project1\Face_Data\s' num2str(i) '\' num2str(j) '.pgm']);
        FaceTestCellArrayData{count}=img;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%PCA data
i = 1; %person 1-25
j = 1; %img 1-10
count = 0;
FacePCACellArrayDataP2 = cell(1, count);

for i = 1:25    
    for j = 1:10
        count = count + 1;  
        img = imread(['\\kc.umkc.edu\kc-users\home\e\emt9q7\My Documents\MATLAB\SupervisedLearning\Project1\Face_Data\s' num2str(i) '\' num2str(j) '.pgm']);
        FacePCACellArrayDataP2{count}=img;
    end
end

%Train
i = 1; %person 26-40
j = 1; %img 1-5
count = 0;
FaceTrainCellArrayDataP2 = cell(1, count);

for i = 26:40    
    for j = 1:5
        count = count + 1;  
        img = imread(['\\kc.umkc.edu\kc-users\home\e\emt9q7\My Documents\MATLAB\SupervisedLearning\Project1\Face_Data\s' num2str(i) '\' num2str(j) '.pgm']);
        FaceTrainCellArrayDataP2{count}=img;
    end
end

%Test
i = 1; %person 26-40
j = 1; %img 6-10
count = 0;
FaceTestCellArrayDataP2 = cell(1, count);

for i = 26:40    
    for j = 6:10
        count = count + 1;  
        img = imread(['\\kc.umkc.edu\kc-users\home\e\emt9q7\My Documents\MATLAB\SupervisedLearning\Project1\Face_Data\s' num2str(i) '\' num2str(j) '.pgm']);
        FaceTestCellArrayDataP2{count}=img;
    end
end
%addpath(fullfile('\\kc.umkc.edu\kc-users\home\e\emt9q7\My Documents\MATLAB\SupervisedLearning\Project1\')); 
% %if you are not me you will have to use a different save path
%save ('FaceCellArrayData');