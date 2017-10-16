% Updated to Matlab 2015
% MC marzo 16
% Introducción de parámetros
clear
close all

disp(' ')
disp('Reduccion de dimensionalidad por seleccion de caracteristicas')
tec_red_dim=input(' Tecnica de seleccion: MDA (0)  PCA (1) ');
switch tec_red_dim
    case 0
        xx_m='mda_clp(X_train,Labels_train,n_clases)';
    case 1
        xx_m='pca(X_train)';
    otherwise
        disp('Escoja una tecnica valida')
    return
end        

i_plot=1;                   %0 NO /1 SI: Representacion inicial de la BD en 3D
i_scplot=1;                 %0 NO /1 SI: Representacion fronteras de decisión en 2D

%% Generacion de la base de datos Gaussiana
sem_aleat=input(' Generacion de datos Gaussianos. Introduzca semilla =');
randn('seed',sem_aleat)
SNR=input('SNR (dB) =');
prac2_gengauss;             % Generacion de BD Gaussiana medias NO alineadas
%prac2_gengauss_al;         % Generacion de BD Gaussiana MEDIAS ALINEADAS
%% Visualizacion de la BD de train.
if i_plot==1
    % SCATTER PLOT
    figure
    gplotmatrix(X_train,X_train,Labels_train,'brm','.',[],'on',[])
    title('Scatter 3D')
    % Representacion en 3D
    figure('name','Plot 3D clusters')
    hold on
   
    Aux=['r','b','k'];
    for i_class=1:n_clases
        index=find(Labels_train==i_class);
        scatter3(X_train(index,1),X_train(index,2),X_train(index,3),Aux(i_class));
    end
    grid
    title('Datos 3D')
    clear i_class Aux
end
clear i_plot

%% Construcción del clasificador para los datos en 3D 
% y calculo de la probabilidad de error

% Datos de entrenamiento
fprintf('\n------- Prob error en 3D -------\n')
fprintf('\n Datos de entrenamiento\n')
linclass = fitcdiscr(X_train,Labels_train);
Linear_out = predict(linclass,X_train);
Linear_Pe_train=sum(Labels_train ~= Linear_out)/length(Labels_train);
fprintf(1,' P(error-LC) = %g   \n', Linear_Pe_train)
quaclass = fitcdiscr(X_train,Labels_train,'discrimType','quadratic');
Quadratic_out= predict(quaclass,X_train);
Quadratic_Pe_train=sum(Labels_train ~= Quadratic_out)/length(Labels_train);
fprintf(1,' P(error-QC) = %g   \n', Quadratic_Pe_train)

% Datos de test
fprintf('\n Datos de test\n')
Linear_out = predict(linclass,X_test);
Linear_Pe_test=sum(Labels_test ~= Linear_out)/length(Labels_test);
fprintf(1,' P(error-LC) = %g   \n', Linear_Pe_test)
Quadratic_out= predict(quaclass,X_test);
Quadratic_Pe_test=sum(Labels_test ~= Quadratic_out)/length(Labels_test);
fprintf(1,' P(error-QC) = %g   \n', Quadratic_Pe_test)

%%  Proyección a 2D
W_fc=eval(xx_m);
W_fc=W_fc(:,1:2);
X_train=X_train*W_fc;
X_test=X_test*W_fc;

% Construcción del clasificador para los datos en 2D
% y calculo de la probabilidad de error

% Datos de entrenamiento
fprintf('\n------- Prob error en 2D -------\n')
fprintf('\n Datos de entrenamiento\n')
linclass = fitcdiscr(X_train,Labels_train);  %LC
Linear_out = predict(linclass,X_train);
Linear_Pe_train=sum(Labels_train ~= Linear_out)/length(Labels_train);
fprintf(1,' P(error-LC) = %g   \n', Linear_Pe_train)
quaclass = fitcdiscr(X_train,Labels_train,'discrimType','quadratic');  %QC
Quadratic_out= predict(quaclass,X_train);
Quadratic_Pe_train=sum(Labels_train ~= Quadratic_out)/length(Labels_train);
fprintf(1,' P(error-QC) = %g   \n', Quadratic_Pe_train)

% Datos de test
fprintf('\n Datos de test\n')
Linear_out = predict(linclass,X_test);  %LC
Linear_Pe_test=sum(Labels_test ~= Linear_out)/length(Labels_test);
fprintf(1,' P(error-LC) = %g   \n', Linear_Pe_test)
Quadratic_out= predict(quaclass,X_test);  %QC
Quadratic_Pe_test=sum(Labels_test ~= Quadratic_out)/length(Labels_test);
fprintf(1,' P(error-QC) = %g   \n', Quadratic_Pe_test)

figure('name','scatter and boundaries')
gscatter(X_train(:,1),X_train(:,2),Labels_train,'krb','ov^',[],'off');
grid
hold on
gscatter(X_test(:,1),X_test(:,2),Labels_test,'krb','ov^',[],'off');
Xmin=min(X_train(:,1));
Xmax=max(X_train(:,1));
Ymin=min(X_train(:,2));
Ymax=max(X_train(:,2));

% Plot the LINEAR classification boundaries for the class 1.
for i_class=2:n_clases
    K = linclass.Coeffs(1,i_class).Const; % retrieve the coefficients for the linear
    L = linclass.Coeffs(1,i_class).Linear;% boundary between the first and second classes
    % Plot the curve K + [x,y]*L  = 0.
    f = @(x1,x2) K + L(1)*x1 + L(2)*x2;
    h = ezplot(f,[Xmin,Xmax,Ymin,Ymax]);
    set(h,'Color','k','LineWidth',2);
end

% Plot the QUADRATIC classification boundaries for the class 1.
for i_class=2:n_clases
    K = quaclass.Coeffs(1,i_class).Const; % retrieve the coefficients for the quadratic
    L = quaclass.Coeffs(1,i_class).Linear;% boundary between the first and second classes
    Q = quaclass.Coeffs(1,i_class).Quadratic;
    % Plot the curve K + [x1,x2]*L + [x1,x2]*Q*[x1,x2]' = 0.
    f = @(x1,x2) K + L(1)*x1 + L(2)*x2 + Q(1,1)*x1.^2 + ...
        (Q(1,2)+Q(2,1))*x1.*x2 + Q(2,2)*x2.^2;
    h = ezplot(f,[Xmin,Xmax,Ymin,Ymax]);
    set(h,'Color','r','LineWidth',2);
end
title('Datos 2D y regiones de decision')
clear Xmin Xmax Ymin Ymax h K L Q V

%%  Proyección a 1D
X_train=X_train(:,1);
X_test=X_test(:,1);

% Construcción del clasificador para los datos en 1D
% y calculo de la probabilidad de error

% Datos de entrenamiento
fprintf('\n------- Prob error en 1D -------\n')
fprintf('\n Datos de entrenamiento\n')
linclass = fitcdiscr(X_train,Labels_train);
Linear_out = predict(linclass,X_train);  %LC
Linear_Pe_train=sum(Labels_train ~= Linear_out)/length(Labels_train);
fprintf(1,' P(error-LC) = %g   \n', Linear_Pe_train)
quaclass = fitcdiscr(X_train,Labels_train,'discrimType','quadratic');  %QC
Quadratic_out= predict(quaclass,X_train);
Quadratic_Pe_train=sum(Labels_train ~= Quadratic_out)/length(Labels_train);
fprintf(1,' P(error-QC) = %g   \n', Quadratic_Pe_train)

% Datos de test
fprintf('\n Datos de test\n')
Linear_out = predict(linclass,X_test);
Linear_Pe_test=sum(Labels_test ~= Linear_out)/length(Labels_test);
fprintf(1,' P(error-LC) = %g   \n', Linear_Pe_test)
Quadratic_out= predict(quaclass,X_test);
Quadratic_Pe_test=sum(Labels_test ~= Quadratic_out)/length(Labels_test);
fprintf(1,' P(error-QC) = %g   \n', Quadratic_Pe_test)

figure('name','scatter and boundaries')
gplotmatrix(X_train,X_train,Labels_train,'brm','.',[],'on',[])
grid
hold on
gplotmatrix(X_test,X_test,Labels_test,'brm','.',[],'on',[])
grid
title('Datos 1D')


