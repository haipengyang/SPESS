close all;clear;clc;rng('shuffle');warning('off');
%% parameter_change
DataIndex = 1; 
runs = 2; 
%%
DataName = {'Yale'};
path = sprintf('%s/dataset.mat',DataName{DataIndex});
load(path);
%%
A = data; k = 8; % cardinality constraint

%% Preprocess data
A(:,find(sum(abs(A),1)==0))=[];% eliminate the zero columns of A
[m,n] = size(A);
if m > n
    [~,S,V]  = svd(A, 'econ');
    sigma_vt = S*V';
    A        = sigma_vt(1:n, :);    
end
tempSum = trace(A'*A); [u,d,v] = svds(A,k); % Run SVD
loss    = norm(A-u*d*v', 'fro')^2;

%% UFS_sparseSS_result
vf = zeros(1,runs); knf = [];popff = zeros(runs,k);
tic;
for i = 1:runs
    display([' time -- ',num2str(i)]);
    [selectedIndex,fitness,kn6,popfit] = UFS_sparseSS(k,A);
    popfit = sort(popfit); popff(i,:) = popfit';
    ErrorRatio = roundn((tempSum-fitness(1))/loss,-3);
    vf(i)      = ErrorRatio; knf = [knf;roundn((tempSum-kn6)/loss,-3)];   
    display(find(selectedIndex==1)); display(ErrorRatio);   
end
toc;
popf6f = mean(popff);popf6f = roundn((tempSum-popf6f)/loss,-3);
kn6f = mean(knf); kn6f = roundn(kn6f,-3); kn6f = kn6f(2:size(kn6f,2)); display(['UFS_sparseSS time: ',num2str(toc/runs)]);
avRunFit6      = num2str(roundn(sum(vf)/runs,-3)); avRunFitStd6 = num2str(roundn(std(vf,1),-3)); avRunTime6 = num2str(roundn(toc/runs,-1));
avRunsparseSS = [avRunFit6,"±",avRunFitStd6,"_",avRunTime6]; display([avRunFit6,'±',avRunFitStd6,'_',avRunTime6]);
