clear; close;
getd = @(p)path(path,p);% Add some directories to the path
getd('../EDAEPRec_demo_ljj/需要比较的图片和mask\');
getd('../EDAEPRec_demo_ljj/需要比较的图片和mask_015\');
getd('../EDAEPRec_demo_ljj/quality_assess\');
getd('../EDAEPRec_demo_ljj/model/');
getd('../EDAEPRec_demo_ljj/ultilies/');
getd('../');

% set to 0 if you want to run on CPU (very slow)
useGPU = 1;
gpuDevice(1);

%% step1 #######%%%%% read sampling %%%%
% line = 30 ;%61  %77  %30; % 
% [mask] = strucrand(256,256,1,line);
% mask = fftshift(fftshift(mask,1),2);
% figure(355); imshow(mask,[]);   %imshow(fftshift(mask),[]);
%% 三种不同的采样方式
%#######%%%%% radial sampling %%%%
load mask_radial90; mask = mask_radial90; 
figure(351); imshow(mask,[]);   %imshow(fftshift(mask),[]);
% %#######%%%%% Cart sampling %%%%
% load mask_cart_085.mat;
% mask = mask_cart_085;
% mask = fftshift(fftshift(mask,1),2);
% figure(352); imshow(mask,[]);   %imshow(fftshift(mask),[]);
% %#######%%%%% random sampling %%%%
% load mask_random015; mask = mask_random015;
% figure(353); imshow(mask,[]);   %imshow(fftshift(mask),[]);

n = size(mask,2);
fprintf(1, 'n=%d, k=%d, Unsamped=%f\n', n, sum(sum(mask)),1-sum(sum(mask))/n/n); %

%imwrite(fftshift(mask), ['mask_77','.png']); 


%% step2 #######%%%%% read test images %%%%
for ImgNo =1:6% 1:9
    for epoch = 30
        load(['.\model_2sigmaP25W20\DnCNN-epoch-' num2str(epoch) '.mat']);
        switch ImgNo
            case 1
                fn1 = 'Leaves256';
                gt = double(imread('Leaves256.tif'));
            case 2
                fn1 ='house'; 
                gt = double(imread('house.tif'));
            case 3
                fn1 ='parrots256';
                gt = double(imread('parrots256.tif'));
            case 4
                fn1 = 'Monarch';
                gt = double(imread('Monarch.png'));
            case 5
                fn1 = 'foreman';
                gt = double(imread('foreman.tif'));
            case 6
                fn1 = 'lena';
                gt = double(imread('lena.tif'));  
        end
    
% gt = double(imread('images/barbara.tif'));
% gt = double(imread('images/boats.tif'));
% gt = double(imread('images/cameraman.tif'));
% gt = double(imread('images/baboon.tif'));
% gt = double(imread('images/peppers.tif'));
% gt = double(imread('images/straw.tif'));

%load lsq28; Img = imrotate(Img, -90); Img(:,end-6:end) = []; Img(:,1:7) = [];
%load lsq68;  Img = imrotate(Img, 90); Img(:,end-6:end) = []; Img(:,1:7) = [];
% load lsq200;  Img = imrotate(Img, 90); Img(:,end-6:end) = []; Img(:,1:7) = [];
% gt = Img;
% gt = 255*gt./max(abs(gt(:)));
figure(334);imshow(abs(gt),[]);
% imwrite(uint8(abs(gt)), ['lsq28','.png']); 


%% step3 #######%%%%% generate K-data %%%%
sigma_d = 0 * 255; %%%
noise = randn(size(gt));
degraded = mask.*(fft2(gt) + noise * sigma_d + (0+1i)*noise * sigma_d); %
Im = ifft2(degraded); 
figure(335);imshow(abs(Im),[]);
 params.gt = gt;
 
%% step4 #######%%%%% run EDAEPRec %%%%
%%% load blind Gaussian denoising model (color image)
modelchannel = 1;
net = dagnn.DagNN.loadobj(net) ;
net.removeLayer('loss') ;
out1 = net.getVarIndex('prediction') ;
net.vars(net.getVarIndex('prediction')).precious = 1 ;
%net.mode = 'test';
%%% move to gpu
if useGPU
    net.move('gpu');
end
params.useGPU = useGPU;
params.net = net;
params.gt = gt;
params.modelchannel = modelchannel;
params.Psigma_net = 25;%%%%%%
params.Wsigma_net = 20;%%%%%%
% run EDAEPRec
[map_Rec,psnr_ssim,psnr_psnr,data_result] = DNCNN_MRIRec5channel_nature_P25W20_v1(Im, degraded, mask, sigma_d, net, params);

[psnr4, ssim4, fsim4, ergas4, sam4] = MSIQA(abs(gt), abs(map_Rec));
[psnr4, ssim4, fsim4, ergas4, sam4]


%% step5 #######%%%%% display Recon result %%%%
figure(666);
subplot(2,3,[4,5,6]);imshow([abs(Im-gt)/255,abs(map_Rec-gt)/255],[]); title('Recon-error');colormap(jet);colorbar;
subplot(2,3,1);imshow(abs(gt)/255); title('Ground-truth');colormap(gray);
subplot(2,3,2);imshow(abs(Im)/255); title('Zero-filled');colormap(gray);
subplot(2,3,3);imshow(abs(map_Rec)/255); title('Net-recon');colormap(gray);
figure(667);imshow([real(gt)/255,imag(gt)/255,abs(gt)/255],[]); 
figure(668);imshow([abs(Im-gt)/255,abs(map_Rec-gt)/255],[]); colormap(jet);colorbar;
% imwrite(uint8(abs(map_Rec)), ['DAEPRec_qx3channel_lsq28_factor10','.png']); 
% imwrite(uint8(abs(map_Rec)), ['DAEPRec_lsq28_factor4','.png']); 
save (['./CS_simgaP25W20_7m25d_90/',fn1 '_' num2str(epoch)],'psnr_psnr','psnr_ssim');
save (['./CS_simgaP25W20_7m25d_90/',fn1 '_' num2str(epoch) '_image'],'data_result');
    end
end