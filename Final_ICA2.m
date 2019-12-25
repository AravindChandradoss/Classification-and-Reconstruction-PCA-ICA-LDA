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

%creating label

data_set=data_set';
data_set=double(data_set);
class_label=[ones(100,1);2*ones(100,1);3*ones(100,1);4*ones(100,1);5*ones(100,1);
    6*ones(100,1);7*ones(100,1);8*ones(100,1);9*ones(100,1);10*ones(100,1);11*ones(100,1);
    12*ones(100,1);13*ones(100,1)];
class_label=double(class_label);




%using all the image

r=10; %fixing the needed components
classl = [13 10 4 1]; % selecting the classes

for j = 1:4
    class = classl(j);
    
    %cropping the selected  data from total data for applying ICA
    X = data_set(class_label(:)==class,:);
    
    %applying kICA and reconstructing the images (for comparison purpose)
    [Zica, W, T, mu]= kICA(X,r);
    Zf1=X*Zica'*Zica;
    
    %reconstruction image loop
    figure
    for i= 1:r
        V_img(:,:,:,i)=uint8(reshape(Zf1(i,:),165,120));
        subplot(2,5,i)
        imshow(V_img(:,:,:,i));
    end
    
    %applying PCA to the generated ICA subspace, ie finding the correlation
    %of the generated subspace and using the PCA subspace to represent the
    %ICA and to reconstuct the images
    
    %applying PCA
    [Zpca, U, mu, eigVecs] = PCA(Zica,5);
    
    %reconstruction using PCA subspace
    Zf1=X*Zpca'*Zpca;
    figure
    for i= 1:r
        V_img(:,:,:,i)=uint8(reshape(Zf1(i,:),165,120));
        subplot(2,5,i)
        imshow(V_img(:,:,:,i));
    end
    
    
    
    
end    


%==============================
%        other codes
%=============================




function [Zica, W, T, mu] = kICA(Z,r)
%
% Syntax:       Zica = kICA(Z,r);
%               [Zica, W, T, mu] = kICA(Z,r);
%               
% Inputs:       Z is an d x n matrix containing n samples of d-dimensional
%               data
%               
%               r is the number of independent components to compute
%               
% Outputs:      Zica is an r x n matrix containing the r independent
%               components - scaled to variance 1 - of the input samples
%               
%               W and T are the ICA transformation matrices such that
%               Zr = T \ W' * Zica + repmat(mu,1,n);
%               is the r-dimensional ICA approximation of Z              
%               mu is the d x 1 sample mean of Z
% Reference:    http://www.cs.nyu.edu/~roweis/kica.html            
% Author:       Brian Moore
%               brimoor@umich.edu            
% Date:         November 12, 2016
%

% Center and whiten data
mu = mean(Z,2);
T = sqrtm(inv(cov(Z')));
Zcw = T * bsxfun(@minus,Z,mu);

% Max-kurtosis ICA
[W, ~, ~] = svd(bsxfun(@times,sum(Zcw.^2,1),Zcw) * Zcw');
Zica = W(1:r,:) * Zcw;

end



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


