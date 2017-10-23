clear;
close all;

%% Options / Initalitation
V_coor=1:256;             % 256 to take all features set 1:256

N_feat=length(V_coor);
% class name: Labels:
% 1(aa);2(ao);3(dcl);4(iy);5(sh);
N_classes=5;
N_fft=256;						%256 (8KHz) 128 (4KHz), 64 (2KHz), 32(1khZ)
%% Database load
load BD_phoneme

%% MEAN IS REMOVED FROM DATABASE
X=X-ones(length(Labels),1)*mean(X);

%% Feature selection
if V_coor(1)~=0
    X=X(:,V_coor);  % Feature selection
end
clear V_coor
 
%% Database partition
P_train=0.7;
Index_train=[];
Index_test=[];
for i_class=1:N_classes
    index=find(Labels==i_class);
    N_i_class=length(index);
    [I_train,I_test] = dividerand(N_i_class,P_train,1-P_train);
    Index_train=[Index_train;index(I_train)];
    Index_test=[Index_test;index(I_test)];
end
% Train Selection
X_train=X(Index_train,:);
Labels_train=Labels(Index_train);
% Test Selection and mixing
X_test=X(Index_test,:);
Labels_test=Labels(Index_test);
clear Index_train Index_test index i_class N_i_class I_train I_test

%% MDA PARTITION

d_max = min(N_classes-1,N_feat);

lc_train_error = zeros(d_max,1);
lc_test_error = zeros(d_max,1);
qc_train_error = zeros(d_max,1);
qc_test_error = zeros(d_max,1);

for d_prima=1:d_max
    mda_vectors = mda_clp(X_train,Labels_train,N_classes);
    W_red = mda_vectors(:,1:d_prima);
    
    X_train_mda = X_train*W_red;
    X_test_mda = X_test*W_red;

    linclass = fitcdiscr(X_train_mda,Labels_train,'prior','empirical');
    quaclass = fitcdiscr(X_train_mda,Labels_train,'discrimType','quadratic','prior','empirical');
   
    Linear_out = predict(linclass,X_train_mda);
    lc_train_error(d_prima)=sum(Labels_train ~= Linear_out)/length(Labels_train);

    Linear_out = predict(linclass,X_test_mda);
    lc_test_error(d_prima)=sum(Labels_test ~= Linear_out)/length(Labels_test);
    
    Quadratic_out= predict(quaclass,X_train_mda);
    qc_train_error(d_prima)=sum(Labels_train ~= Quadratic_out)/length(Labels_train);

    Quadratic_out= predict(quaclass,X_test_mda);
    qc_test_error(d_prima)=sum(Labels_test ~= Quadratic_out)/length(Labels_test);
    
end

%% Create error plot

figure;
hold on;
plot(lc_train_error);
plot(lc_test_error);
plot(qc_train_error);
plot(qc_test_error);
legend('LC Train Error', 'LC Test Error', 'QC Train Error', 'QC Test Error');
