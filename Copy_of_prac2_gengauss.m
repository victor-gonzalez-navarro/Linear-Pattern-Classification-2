% prac2_gengauss
% Se generan 3 gaussianas NO alineadas
% La matriz de covarianza es distinta entre gaussianas
%% Parámetros de configuración
n_clases=3;
n_muestras=[500;500;500];       % Numero de muestras para cada clase
n_feat=3;
dist=1;                         % Inter-Symbol Distance
M_Means=dist*[1,1,-1;-1,0,-1;-1,-1,0]; 	%Vector de Medias
% Calculo energia
Energia=0;
for i_clase=1:n_clases
   V=squeeze(M_Means(i_clase,:));
   Energia=Energia+V*V';
end
Energia=Energia/i_clase;
%Calculo sigma
SNR=10^(SNR/10);
sig=Energia/SNR;
sig=sig/n_feat;
clear Energia V dist
%Variable tridimensional para la matriz de covarianza
M_cov=zeros(n_feat,n_feat,n_clases);
D=sig*[1 1 1;0.1 0.3 0.6;2 0.01 0.99];  % Autovalores asociados a las direcciones
                                        % principales de cada una de las clases
for i_clase=1:n_clases
   H=randn(n_feat);                      % Generacion de autovectores asociados a las
   [U, ~]=eig(H*H');                    % direcciones principales para cada clase   
   M_cov(:,:,i_clase)=U*diag(D(i_clase,:))*U';%Matriz de covarianzas clase i_clase
end
clear H U i_clase sig

%% Generación del dataset de entrenamiento gaussiano
X_train=[];
Labels_train=[];
for i_clase=1:n_clases
    X_train=[X_train;mvnrnd(M_Means(i_clase,:),M_cov(:,:,i_clase),n_muestras(i_clase))];
    Labels_train=[Labels_train; i_clase*ones(n_muestras(i_clase),1)];
end

%% Generación del dataset de test gaussiano
X_test=[];
Labels_test=[];
for i_clase=1:n_clases
    X_test=[X_test;mvnrnd(M_Means(i_clase,:),M_cov(:,:,i_clase),round(0.25*n_muestras(i_clase)))];
    Labels_test=[Labels_test; i_clase*ones(round(0.25*n_muestras(i_clase)),1)];
end
clear i_clase
