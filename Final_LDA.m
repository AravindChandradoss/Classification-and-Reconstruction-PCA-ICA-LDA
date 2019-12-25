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
      imageArray = imresize(imageArray, 0.3);
      data_set(:,k+(l-1)*length(bmpFiles))=reshape(imageArray,1,50*36);
    end
end

%creating labels
data_set=data_set';
data_set=double(data_set);
class_label=[ones(100,1);2*ones(100,1);3*ones(100,1);4*ones(100,1);5*ones(100,1);
    6*ones(100,1);7*ones(100,1);8*ones(100,1);9*ones(100,1);10*ones(100,1);11*ones(100,1);
    12*ones(100,1);13*ones(100,1)];
class_label=double(class_label);

%applying LDA, using the funstion mylda(...)
% data_set is the image data 
% class_label is the corresponding label for each image
[sLDA, WLDA, M, WPCA]=mylda(data_set,class_label,10);

%reconstructing the images in loop
r=1300;
figure
for i= 1:100:r
    %reconstruction of image by using PCA and LDA space
    %In this function, PCA is applied before LDA and thus, while
    %reconstruction, we have to make sure that reconstruct the image form
    %the PCA too. i.e. LDA is followed by PCA for reconstruction
    
    LDAspace=(data_set(i,:)-M)*WPCA*WLDA*WLDA'*WPCA'+M;
    
    %reshaping and changing datatype
    V_img(:,:,:,i)=uint8(reshape(LDAspace,50,36));
    
    subplot(5,3,ceil(i/100))
    imshow(V_img(:,:,:,i));
end


% ====================================
% ====================================
% %tried to project and reconstruct unseen image using LDA space
% %make sure to have any one of images from AR_dataset and rename it 
% %to "1.bmp"
% % un comment to run the following part
% 
% Zimg=imread('1.bmp');
% Zimg=rgb2gray(Zimg);
% Zimg=imresize(Zimg,0.3);
% Z_vec=reshape(Zimg,1,50*36);
% Zvec=double(Z_vec);
% figure
% for i= 1:1
%     LDAspace=(Zvec(i,:)-M)*WPCA*WLDA*WLDA'*WPCA'+M;
%     V_img(:,:,:,i)=uint8(reshape(LDAspace,50,36));
%     %subplot(2,5,i)
%     subplot(5,3,ceil(i/100))
%     imshow(V_img(:,:,:,i));
% end


%====================================================
%============== Other codes =========================
%====================================================
%====================================================
function [sLDA, WLDA, M, WPCA]=mylda(data,class,n)
% [sLDA WLDA M WPCA]=mylda(data,class,n)
% this function written by muhammet balcilar
% yildiz technical university computer engineering department
% istanbul turkiye 2011

% this function convert data from its original space to LDA space
% if number of data samples is less than number of diamension, PCA is
% implemented for reducing number of diamension to #samples-1. 
% after PCA, LDA is implemented for reducing diamention to n.

% data is consist of M rows(sample size), N cols(dimensions)
% class is consist of M rows(sample size), 1 cols , each element of class 
% is shows class number of each data sample 
% (class number must be integer 1 to classsize)
% n is the number of outputs data diamensions.(optionally)
% sLDA is consist of M rows(sample size) n cols(new dimensions)
% WPCA is translate matrix which convert to original space to PCA space
% M is the mean vector of training set
% WLDA is the translate matrix which convert to original space to LDA space
% exaple: there are 4 samples which have 5 diamensions.first two samples
% are member of class 1 others are member of class 2.
% Train= [5.6,5.7,5.5,5.7 5.6;
%     5.7,5.3,5.1,5.0 5.2;
%     10.6,9.9,10.4,10.7 10.2;
%     10.7,9.8,9.9,10 10];
% Class=[1;1;2;2];
% [sLDA WLDA M WPCA]=mylda(Train,Class)
% Test= [4.9 5.5 4.8 5.7 5];
% LDATEST = (Test-M)*WPCA*WLDA

    usinif=unique(class);
    if nargin==2
        n=length(usinif)-1;
    end

    if size(data,2)>=size(data,1)
        % PCA start
        O=data';
        m=(mean(O'))';
        for i=1:size(O,2)
            mO(:,i)=O(:,i)-m;
        end
        CV=mO*mO';
        [v u]=eig(CV);
        D=v(:,end-size(data,1)+2:end); 
        yO=(mO'*D)';
        M=m';
        WPCA=D;
        % PCA finished
    else
        yO=data';
        M=zeros(1,size(data,2));
        WPCA=1;    
    end


    % LDA start
    mU=(mean(yO'))';
    mK=[];
    for i=1:length(usinif)
        I=find(class==i);
        ort=(mean(yO(:,I)'))';
        mK=[mK ort];
        for j=1:length(I)
            UU(:,I(j))=yO(:,I(j))-ort;
        end
    end
    for i=1:length(usinif)
        I=find(class==i);
        S{i}= UU(:,I)*UU(:,I)';
    end
    SW=S{1};
    for i=2:length(usinif)
        SW=SW+S{i};
    end

    for i=1:length(usinif)
        mmK(:,i)=mK(:,i)-mU;
    end
    
    SB=2*mmK*mmK';
    [w u]=eig(SB,SW);
    u=abs(diag(u));
    u=[u [1:length(u)]'];
    u=sortrows(u,1);
    WLDA=w(:,u(end-n+1:end,2)); 
    sLDA=(yO'*WLDA)';
end


