clc
clear all
close all 

%load data
load Mini_project#1/featureMat_Latex_test_bioLBP.mat
load Mini_project#1/featureMat_Latex_train_bioLBP.mat
load Mini_project#1/featureMat_liv_test_bioLBP.mat
load Mini_project#1/featureMat_liv_train_bioLBP.mat

%train data
train_live_features = featureMat_liv_train_bioLBP;
train_fake_features = featureMat_Latex_train_bioLBP;

%concatenate features to create training data
train = [train_live_features' train_fake_features'];

%define labels 1000(1's for live) 200(0's for fakes)
labels = cat(2, ones(1,1000), zeros(1,200));

%Train Naive Bayes classifier --- latex
prior = [0.6 0.4];
model = fitcnb(train', labels', 'Prior', prior);



%-------------------TEST--------------------------------------

%test data
test_live_features = featureMat_liv_test_bioLBP;
test_fake_features = featureMat_Latex_test_bioLBP;

test = [test_live_features' test_fake_features'];


%Compute loss and Resubstitution error
%Calculate Loss 
loss = loss(model,test',labels') %Test the classifier on test set? 

%Resubstitution Error
resubError = resubLoss(model)






