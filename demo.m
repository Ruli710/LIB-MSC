clc
clear
addpath('dataset');addpath('other function');
load('Mfeat.mat');
% 超参数设置
lambda = 10;   % 低秩项权重
p = 0.5;       % Schatten p-范数参数
beta = 2;      % 信息保留项权重
gamma = 1;     % 冗余抑制项权重
%-------------parameterization setup--------------
%       lambda  , p    ,  beta , gamma     ,  mu
% yaleB:    1   , 0.5  ,  2    ,   1	   , 1.00E-05
% COIL20:   10  , 0.5  ,  1    ,   1	   , 1.00E-05
% Handwritten4:10  ，0.5 ，1   ,   3	   , 1.00E-02
% UCI:      10  , 0.5  ,   2   ,  1e-5	   , 1.00E-02
% WikipediaArticles:10 , 0.5 , 1 , 3       , 1.00E-02
% Mfeat:    10  , 0.5  ,  2    ,   1	   , 1.00E-02
% BBC:      10  , 0.6  ,  10   ,   1	   , 1.00E-02
% ------------------------------------------------
for i=1:size(X,2)
    X{i} = X{i}./(repmat(sqrt(sum(X{i}.^2,1)),size(X{i},1),1)+eps);
end
nCluster = length(unique(Y));
mu=1e-2;
rho=2.8;
tic
idx = 1;
for Index = 1 : length(beta)
        betaTemp = beta(Index);
        for gammaIndex = 1 : length(gamma)
            gammaTemp = gamma(gammaIndex);
            % Main algorithm
            fprintf('Please wait a few minutes\n');
            Zn = LIB_MSC(X, lambda, mu, rho, p, betaTemp, gammaTemp);
            M=retain(Zn);
            W=postprocessor(M);
            nCluster = length(unique(Y));
            label = new_spectral_clustering(W,nCluster);
            result = Clustering8Measure(Y, label);
            disp(['acc   ' num2str(result(1)*100) '%     '  '  nmi   ' num2str(result(2)*100) '%     ' 'Fscore   ' num2str(result(3)*100) '%'  'Precision   ' num2str(result(4)*100) '%     ' '  AR   ' num2str(result(5)*100) '%     ''     time     '   num2str(toc)])
            disp(betaTemp)
            disp(gammaTemp)
        end  
end



