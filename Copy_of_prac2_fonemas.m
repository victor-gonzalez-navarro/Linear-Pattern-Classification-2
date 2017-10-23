% Updated to Matlab2015
% MC Febrero 2016
clear;
close all;  % close all previous figures

%% Options / Initalitation
i_dib=1;				 % 0 NO /1 YES: plot spectrums
V_coor=1:64;             % 256 to take all features set 1:256 
%V_coor=[18 63];         % EXAMPLE: Selection of a subset of two features [25, 60]

N_feat=length(V_coor);
% class name: Labels:
% 1(aa);2(ao);3(dcl);4(iy);5(sh);
N_classes=5;
N_fft=256;						%256 (8KHz) 128 (4KHz), 64 (2KHz), 32(1khZ)
%% Database load
load BD_phoneme

%% MEAN IS REMOVED FROM DATABASE
X=X-ones(length(Labels),1)*mean(X);

%% Spectrum plot
if i_dib==1
    Frec_max=8*N_fft/256;			%Max frequency in KHz
    eje_frec=(0:N_fft-1)*Frec_max/N_fft;
    clases=['aa';'ao';'dc';'iy';'sh'];
    figure('name','LOG(Espectrum)')
    for i_clas=1:N_classes
        subplot(3,2,i_clas)
        hold on
        index=find(Labels==i_clas);
        for i1=1:length(index)
            plot(eje_frec,X(index(i1),1:N_fft));
        end
        hold off
        grid
        zoom on
        xlabel('frec(KHz)')
        ylabel(clases(i_clas,:));
    end
    subplot(3,2,N_classes+1)
    hold on
    i_color=['b' 'r' 'g' 'k' 'y'];
    for i_clas=1:N_classes
        index= Labels==i_clas;
        aux=mean(X(index,1:N_fft));
        plot(aux,i_color(i_clas));
    end
    hold off
    grid
    zoom on
    xlabel('Feature Number')
    ylabel('log espectro')
    title('Average');
    clear index aux i_color i_clas eje_frec Frec_max
end
clear i_dib N_fft

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

%% Create a default (linear) discriminant analysis classifier:
linclass = fitcdiscr(X_train,Labels_train,'prior','empirical')
Linear_out = predict(linclass,X_train);
Linear_Pe_train=sum(Labels_train ~= Linear_out)/length(Labels_train);
fprintf(1,' error Linear train = %g   \n', Linear_Pe_train)
Linear_out = predict(linclass,X_test);
Linear_Pe_test=sum(Labels_test ~= Linear_out)/length(Labels_test);
fprintf(1,' error Linear test = %g   \n', Linear_Pe_test)

%% Create a quadratic discriminant analysis classifier:
quaclass = fitcdiscr(X_train,Labels_train,'discrimType','quadratic','prior','empirical')
Quadratic_out= predict(quaclass,X_train);
Quadratic_Pe_train=sum(Labels_train ~= Quadratic_out)/length(Labels_train);
fprintf(1,' error Quadratic train = %g   \n', Quadratic_Pe_train)
Quadratic_out= predict(quaclass,X_test);
Quadratic_Pe_test=sum(Labels_test ~= Quadratic_out)/length(Labels_test);
fprintf(1,' error Quadratic test = %g   \n', Quadratic_Pe_test)

%% Test confusion matrices
CM_Linear_test=confusionmat(Labels_test,Linear_out)
CM_Quadratic_test=confusionmat(Labels_test,Quadratic_out)

%% create a scatter plot of the data
if N_feat==2
    figure('name','scatter and boundaries for class aa')
    gscatter(X(:,1),X(:,2),Labels,'krbgy','ov^*+')
    grid
    hold on
    Xmin=min(X(:,1));
    Xmax=max(X(:,1));
    Ymin=min(X(:,2));
    Ymax=max(X(:,2));
 
    % Plot the LINEAR classification boundaries for the class 1.
    for i_class=2:N_classes
        K = linclass.Coeffs(1,i_class).Const; % retrieve the coefficients for the linear
        L = linclass.Coeffs(1,i_class).Linear;% boundary between the first and second classes
        % Plot the curve K + [x,y]*L  = 0.
        f = @(x1,x2) K + L(1)*x1 + L(2)*x2;
        h = ezplot(f,[Xmin,Xmax,Ymin,Ymax]);
        h.Color = 'k';
        h.LineWidth = 2;
    end
    
    % Plot the QUADRATIC classification boundaries for the class 1.
    for i_class=2:N_classes
        K = quaclass.Coeffs(1,i_class).Const; % retrieve the coefficients for the quadratic
        L = quaclass.Coeffs(1,i_class).Linear;% boundary between the first and second classes
        Q = quaclass.Coeffs(1,i_class).Quadratic;
        % Plot the curve K + [x1,x2]*L + [x1,x2]*Q*[x1,x2]' = 0.
        f = @(x1,x2) K + L(1)*x1 + L(2)*x2 + Q(1,1)*x1.^2 + ...
            (Q(1,2)+Q(2,1))*x1.*x2 + Q(2,2)*x2.^2;
        h = ezplot(f,[Xmin,Xmax,Ymin,Ymax]);
        h.Color = 'r';
        h.LineWidth = 2;
    end
    clear Xmin Xmax Ymin Ymax h K L Q V
    title('LC Boundaries (k) QC Boundaries (r)')
end

