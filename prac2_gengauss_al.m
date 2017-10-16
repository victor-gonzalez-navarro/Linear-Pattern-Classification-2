% prac2_gengauss_al
% Se generan 3 gaussianas de centros alineados
% Se conserva varianza pequeña en la dirección de alineación
%% Parámetros de configuración
n_clases=3;
n_muestras=[500;500;500];       % Numero de muestras para cada clase
n_feat=3;
dist=1;                         % Inter-Symbol Distance
M_Means=dist*[1,0,0;0,0,0;-1,0,0]; 	%Vector de Medias
% % Calculo energia
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
Vsig2=[0.1 10 10];
Vsig2=3*Vsig2/norm(Vsig2);
Vsig2=sig*Vsig2; % Autovalores asociados a las direcciones
H=randn(n_feat);                      % Generacion de autovectores asociados a las
[U, ~]=eig(H*H');                    % direcciones principales para cada clase
M_Means=M_Means*U;
for i_clase=1:n_clases
    M_cov(:,:,i_clase)=U'*diag(Vsig2)*U;%Matriz de covarianzas clase i_clase
end
clear H U i_clase Vsig2 sig

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
