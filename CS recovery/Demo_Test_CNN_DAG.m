%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% @article{zhang2017beyond,
%   title={Beyond a {Gaussian} denoiser: Residual learning of deep {CNN} for image denoising},
%   author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
%   journal={IEEE Transactions on Image Processing},
%   year={2017},
%   volume={26}, 
%   number={7}, 
%   pages={3142-3155}, 
% }

% by Kai Zhang (1/2018)
% cskaizhang@gmail.com
% https://github.com/cszn
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% clear; clc;

%% testing set
addpath(fullfile('utilities'));

folderModel = 'model_sigma20';
folderTest  = '../testsets';
folderResult= 'results';
imageSets   = {'BSD68','Set12'}; % testing datasets
setTestCur  = imageSets{2};      % current testing dataset


showresult  = 1;
gpu         = 1;


PnoiseSigma  = 25;
WnoiseSigma  = 20;

% load model
epoch       = 4; %9;

modelName   = 'DnCNN';

% case one: for the model in 'data/model'
load(fullfile('model_2sigmaP25W20',[modelName,'-epoch-',num2str(epoch),'.mat']));

% case two: for the model in 'utilities'
% load(fullfile('utilities',[modelName,'-epoch-',num2str(epoch),'.mat']));


net = dagnn.DagNN.loadobj(net) ;

net.removeLayer('loss') ;
out1 = net.getVarIndex('prediction') ;
net.vars(net.getVarIndex('prediction')).precious = 1 ;

net.mode = 'test';

if gpu
    net.move('gpu');
end

% read images
ext         =  {'*.jpg','*.png','*.bmp'};
filePaths   =  [];
for i = 1 : length(ext)
    filePaths = cat(1,filePaths, dir(fullfile(folderTest,setTestCur,ext{i})));
end

folderResultCur       =  fullfile(folderResult, [setTestCur,'_',int2str(2520)]);
if ~isdir(folderResultCur)
    mkdir(folderResultCur)
end


% PSNR and SSIM
PSNRs = zeros(1,length(filePaths));
SSIMs = zeros(1,length(filePaths));


for i = 1 : length(filePaths)
    
    % read image
    label = imread(fullfile(folderTest,setTestCur,filePaths(i).name));
    [~,nameCur,extCur] = fileparts(filePaths(i).name);
    [w,h,c]=size(label);
    if c==3
        label = rgb2gray(label);
    end
    
    % add additive Gaussian noise
    randn('seed',0);
    input = im2single(label);
    Ysize = size(input,1)+1; % +3;
    Xsize = size(input,2)+1; % +3;
    input1 = zeros(Ysize,Xsize,5,1,'single');
    addnoise = zeros(Ysize,Xsize,5,1,'single');
    addnoise(:,:,1,:) = PnoiseSigma/255*randn(size(input1(:,:,1,:)),'single');
    addnoise(:,:,2:5,:) = WnoiseSigma/255*randn(size(input1(:,:,2:5,:)),'single');
    [ld,hd,lr,hr] = wfilters('bior1.1');
    input1(2:end,2:end,1) = input;
    [Wave_S,Wave_HW,Wave_WH,Wave_WW,etl] = ocwt2dliu1(input,ld,hd,1);
    input1(:,:,2) = Wave_S;input1(:,:,3) = Wave_HW{1,1};input1(:,:,4) = Wave_WH{1,1};input1(:,:,5) = Wave_WW{1,1};
    input1 = input1 + addnoise;
    input1 = single(input1);
    if gpu
        input1 = gpuArray(input1);
    end
    net.eval({'input', input1}) ;
    % output (single)
    output = gather(input1) - gather(squeeze(gather(net.vars(out1).value)));
    Wave_S = output(:,:,2); Wave_HW{1,1} = output(:,:,3); Wave_WH{1,1} = output(:,:,4); Wave_WW{1,1} = output(:,:,5);
    output11 = iocwt2dliu1(Wave_S,Wave_HW,Wave_WH,Wave_WW,etl,lr,hr);
    output_mean = (output11 + squeeze(output(2:end,2:end,1)))/2;
    
    % calculate PSNR and SSIM
    [PSNRCur, SSIMCur] = Cal_PSNRSSIM(label,im2uint8(output_mean),0,0);
    if showresult
        figure(10+i);imshow(cat(2,im2uint8(label),im2uint8(output_mean)));
        title([filePaths(i).name,'    ',num2str(PSNRCur,'%2.2f'),'dB','    ',num2str(SSIMCur,'%2.4f')])
%         imwrite(im2uint8(output), fullfile(folderResultCur, [nameCur, '_' int2str(noiseSigma),'_PSNR_',num2str(PSNRCur*100,'%4.0f'), extCur] ));
        drawnow;
       % pause()
    end
    PSNRCur
    SSIMCur
    PSNRs(i) = PSNRCur;
    SSIMs(i) = SSIMCur;
end


disp([mean(PSNRs),mean(SSIMs)]);




