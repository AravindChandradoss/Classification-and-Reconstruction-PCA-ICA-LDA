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




%using all the image
r=10; %no of components
classl = [13 10 4 1]; %selecting classes
for j = 1:4
    class = classl(j);
    %cropping the needed class to perform ICA
    X = data_set(class_label(:)==class,:);
    
    %applying ICA using the function kICA()
    [Zica, W, T, mu]= kICA(X,r);
    
    %Reconstructing the Zica to realworld space  for visualizing purpose
    Zf1=X*Zica'*Zica;
    
    %the first ten vector imgages were reshaped and displayed (for four
    %selected classes)
    figure
    for i= 1:r
        V_img(:,:,:,i)=uint8(reshape(Zf1(i,:),165,120));
        subplot(2,5,i)
        imshow(V_img(:,:,:,i));
    end
    
   
% % The abouve methode is based on kurtosis. and there are other methode too.    
% % Below is a similar apporach, called FastICA
% % Plz, Make sure to test k-ICA first, and then check Fast-ICA
% % I think, kICA is better than fastICA, as it has more details and Fast
% % ICA has more region dark (was like disturbed by other features)
% % Uncomment the following part, if you want to see FastICA Result
%     [Zica, W, T, mu]= fastICA(X,r);
%     Zf1=X*Zica'*Zica;
%    
%     figure
%     for i= 1:r
%         V_img(:,:,:,i)=uint8(reshape(Zf1(i,:),165,120));
%         subplot(2,5,i)
%         imshow(V_img(:,:,:,i));
%    end
% 

end    



% % I also tried to project and reconstruct an unseen image using the Zica.
% % the results are as followes
% 
% %plz make sure you have any one of the image form AR_dataset to run the
% %following part
% % uncomment the following to test it result.
% 
% Zimg=imread('1.bmp');
% Zimg=rgb2gray(Zimg);
% Z_vec=reshape(Zimg,1,165*120);
% Zvec=double(Z_vec);
% 
% r=10;
% for j = 1:4
%     class= classl(j);
%     X = data_set(class_label(:)==class,:);
%     [Zica, W, T, mu]= kICA(X,r);  
%     figure
%     for i= 1:1
% 
%         Zf=Zvec*Zica(i,:)'*Zica(i,:);
%         V_img(:,:,:,i)=uint8(reshape(Zf,165,120));
%         imshow(V_img(:,:,:,i));
%     end
%     
% end


%===============================
%      other code
%===============================


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


function [Zica, W, T, mu] = fastICA(Z,r,type,flag)
%
% Syntax:       Zica = fastICA(Z,r);
%               Zica = fastICA(Z,r,type);
%               Zica = fastICA(Z,r,type,flag);
%               [Zica, W, T, mu] = fastICA(Z,r);
%               [Zica, W, T, mu] = fastICA(Z,r,type);
%               [Zica, W, T, mu] = fastICA(Z,r,type,flag);
%               
% Inputs:       Z is an d x n matrix containing n samples of d-dimensional
%               data
%               
%               r is the number of independent components to compute
%               
%               [OPTIONAL] type = {'kurtosis','negentropy'} specifies
%               which flavor of non-Gaussianity to maximize. The default
%               value is type = 'kurtosis'
%               
%               [OPTIONAL] flag determines what status updates to print
%               to the command window. The choices are
%                   
%                       flag = 0: no printing
%                       flag = 1: print iteration status
%               
% Outputs:      Zica is an r x n matrix containing the r independent
%               components - scaled to variance 1 - of the input samples
%               
%               W and T are the ICA transformation matrices such that
%               Zr = T \ W' * Zica + repmat(mu,1,n);
%               is the r-dimensional ICA approximation of Z
%               
%               mu is the d x 1 sample mean of Z
% Reference:    Hyvrinen, Aapo, and Erkki Oja. "Independent component
%               analysis: algorithms and applications." Neural networks
%               13.4 (2000): 411-43               
% Author:       Brian Moore
%               brimoor@umich.edu              
% Date:         April 26, 2015
%               November 12, 2016
%               May 4, 2018

% Constants
TOL = 1e-6;         % Convergence criteria
MAX_ITERS = 100;    % Max # iterations

% Parse inputs
if ~exist('flag','var') || isempty(flag)
    % Default display flag
    flag = 1;
end
if ~exist('type','var') || isempty(type)
    % Default type
    type = 'kurtosis';
end
n = size(Z,2);

% Set algorithm type
if strncmpi(type,'kurtosis',1)
    % Kurtosis
    USE_KURTOSIS = true;
    algoStr = 'kurtosis';
elseif strncmpi(type,'negentropy',1)
    % Negentropy
    USE_KURTOSIS = false;
    algoStr = 'negentropy';
else
    % Unsupported type
    error('Unsupported type ''%s''',type);
end

% Center and whiten data
[Zc, mu] = centerRows(Z);
[Zcw, T] = whitenRows(Zc);

% Normalize rows to unit norm
normRows = @(X) bsxfun(@rdivide,X,sqrt(sum(X.^2,2)));

% Perform Fast ICA
if flag
    % Prepare status updates
    fmt = sprintf('%%0%dd',ceil(log10(MAX_ITERS + 1)));
    str = sprintf('Iter %s: max(1 - |<w%s, w%s>|) = %%.4g\\n',fmt,fmt,fmt);
    fprintf('***** Fast ICA (%s) *****\n',algoStr);
end
W = normRows(rand(r,size(Z,1))); % Random initial weights
k = 0;
delta = inf;
while delta > TOL && k < MAX_ITERS
    k = k + 1;
    
    % Update weights
    Wlast = W; % Save last weights
    Sk = W * Zcw;
    if USE_KURTOSIS
        % Kurtosis
        G = 4 * Sk.^3;
        Gp = 12 * Sk.^2;
    else
        % Negentropy
        G = Sk .* exp(-0.5 * Sk.^2);
        Gp = (1 - Sk.^2) .* exp(-0.5 * Sk.^2);
    end
    W = (G * Zcw') / n - bsxfun(@times,mean(Gp,2),W);
    W = normRows(W);
    
    % Decorrelate weights
    [U, S, ~] = svd(W,'econ');
    W = U * diag(1 ./ diag(S)) * U' * W;
    
    % Update convergence criteria
    delta = max(1 - abs(dot(W,Wlast,2)));
    if flag
        fprintf(str,k,k,k - 1,delta);
    end
end
if flag
    fprintf('\n');
end

% Independent components
Zica = W * Zcw;

end
function [Zc, mu] = centerRows(Z)
% Compute sample mean
mu = mean(Z,2);

% Subtract mean
Zc = bsxfun(@minus,Z,mu);
end

function [Zw, T] = whitenRows(Z)

% Compute sample covariance
R = cov(Z');

% Whiten data
[U, S, ~] = svd(R,'econ');
T  = U * diag(1 ./ sqrt(diag(S))) * U';
Zw = T * Z;
end


