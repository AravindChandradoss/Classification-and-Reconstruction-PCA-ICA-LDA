clear all;
close all;

%Make sure to run the code from the director where the datafolder is
%present ( ie the folder which has "AR_database_cropped")
%getting files and arranging them

path=pwd;
dataFolder = strcat(pwd,'/AR_database_cropped/test2');
if ~isdir(dataFolder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', dataFolder);
  uiwait(warndlg(errorMessage));
  return;
end

for l=1:1:13
    filePattern = fullfile(dataFolder, strcat('*-',string(sprintfc('%02d',l)),'.bmp'));
    bmpFiles = dir(filePattern);
    for k = 1:length(bmpFiles)
      baseFileName = bmpFiles(k).name;
      fullFileName = fullfile(dataFolder, baseFileName);
      fprintf(1, 'Now reading %s\n', fullFileName);
      imageArray = imread(fullFileName);
      imageArray = rgb2gray(imageArray);
      data_set(:,k+(l-1)*length(bmpFiles))=reshape(imageArray,1,165*120);
    end
end

%creating labels
data_set=data_set';
data_set=double(data_set);
class_label=[ones(100,1);2*ones(100,1);3*ones(100,1);4*ones(100,1);5*ones(100,1);
    6*ones(100,1);7*ones(100,1);8*ones(100,1);9*ones(100,1);10*ones(100,1);11*ones(100,1);
    12*ones(100,1);13*ones(100,1)];
class_label=double(class_label);

r=55;
classl = [13 10 4 1];
for j = 1:1:4
    class = classl(j);
    X = data_set(class_label(:)==class,:);
    
    %applying PCA for dataset
    [Zpca, U, mu, eigVecs]=PCA(X,r);
    
    %Again...applying PCA for the generated subset
    [Zpca2, U2, mu2, eigVecs2]=PCA(Zpca,15);    
    
    %reconstruting the image from subspace using PCA (Zpca1)
    %(for comparision purpose)
    Zr=U*Zpca + repmat(mu,1,19800);
    
    %reconstruting the image from subspace using PCA (Zpca2)
    Zr4=U*U2*Zpca2 + repmat(mu,1,19800);
    
    
    %changing datatype and reshaping for visualization
    recimg=uint8(Zr(class,:));
    recimg4=uint8(Zr4(class,:));    
    recimg=reshape(recimg,165,120);
    recimg4=reshape(recimg4,165,120);
    
    
    figure
    imshow(recimg);
    figure
    imshow(recimg4);
end

% The following are the funcitons that I have used for this code.

function [Zpca, U, mu, eigVecs] = PCA(Z,r)
%
% Syntax:       Zpca = PCA(Z,r);
%               [Zpca, U, mu] = PCA(Z,r);
%               [Zpca, U, mu, eigVecs] = PCA(Z,r);
%               
% Inputs:       Z is an d x n matrix containing n samples of d-dimensional
%               data
%               
%               r is the number of principal components to compute
%               
% Outputs:      Zpca is an r x n matrix containing the r principal
%               components - scaled to variance 1 - of the input samples
%               
%               U is a d x r matrix of coefficients such that
%               Zr = U * Zpca + repmat(mu,1,n);
%               is the r-dimensional PCA approximation of Z
%               
%               mu is the d x 1 sample mean of Z
%               
%               eigVecs is a d x r matrix containing the scaled
%               eigenvectors of the sample covariance of Z
%               
% Description:  Performs principal component analysis (PCA) on the input
%               data
%               
% Author:       Brian Moore
%               brimoor@umich.edu
%               
% Date:         April 26, 2015
%               November 7, 2016
%

% Center data
mu = mean(Z,2);
Zc = bsxfun(@minus,Z,mu);

% Compute truncated SVD
%[U, S, V] = svds(Zc,r); % Equivalent, but usually slower than svd()
[U, S, V] = svd(Zc,'econ');
U = U(:,1:r);
S = S(1:r,1:r);
V = V(:,1:r);

% Compute principal components
Zpca = S * V';
%Zpca = U' * Zc; % Equivalent but slower

    if nargout >= 4
        % Scaled eigenvectors
        eigVecs = bsxfun(@times,U,diag(S)' / sqrt(size(Z,2)));
    end
end
