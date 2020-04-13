% close all;
% clear all;clc;
% add MatCaffe path
% addpath ../mnt/data/siavash/caffe/matlab;
getd = @(p)path(path,p);% Add some directories to the path
% getd('../traindata_lsq\');
% getd('../quality_assess\');
% getd('../');
% getd('../DAEP_diffSigma\');
getd('image/')
% addpath(genpath('D:\ZJJ\DAEP_MWCNN'));
% set to 0 if you want to run on CPU (very slow)

useGPU = 1;
gpuDevice(1);
%% Deblurring demo
% load image and kernel
load('kernels.mat');
%%��ǰDAEP_% gt = double(imread('straw.tif'));
% gt=gt(1:128,1:128);
% gt = double(imread('101085.jpg'));
% gt = double(imread('qqshow_1119804713.jpg'));
%% ���ڵ�DAEP_MWCNN
for ImgNo =2:6% 1:9
    switch ImgNo
        case 1
            fn1 = 'baboon';
            fn = double(imread('baboon.tif'));
        case 2
            fn1 = 'straw';
            fn = double(imread('straw.tif'));
        case 3
            fn1 = 'cameraman';
            fn = double(imread('cameraman.tif'));
        case 4
            fn1 = 'Peppers';
            fn = double(imread('Peppers.bmp'));
        case 5
            fn1 = 'boats';
            fn =double(imread('boats.tif'));
        case 6
            fn1 = 'Barbara256';
            fn = double(imread('Barbara256.tif'));
            
    end
    
    for kernel_a = 1:3
        switch kernel_a
            case 1
                aa =1313;
                kernel = kernels{5};
            case 2
                aa =1919;
                kernel = kernels{1};
            case 3
                aa =2525;
                kernel = kernels{7};
        end
        for alpha =  1:2
            switch alpha
                case 1
                    a1 = 1;
                    a = 0.01;
                case 2
                    a1 = 3;
                    a = 0.03;
            end
            
            gt = fn;
            w = size(gt,2); w = w - mod(w, 2);
            h = size(gt,1); h = h - mod(h, 2);
            gt_extend = double(gt(1:h, 1:w, :)); % for some reason Caffe input needs even dimensions...
            %gt��ԭͼ����0��
            
            
            % kernel = kernels{1};%19%17%15%27%13%21%23%%25
            
            sigma_d = 255 * a;
            pad = floor(size(kernel)/2);   %% �������Ҹĵ� floor(size(kernel)/2);
            %y = floor(x) ������x��Ԫ��ȡ����ֵyΪ�����ڱ����������������ڸ������ֱ��ʵ�����鲿ȡ��
            gt_extend = padarray(gt, pad, 'replicate', 'both');
            %��both���ڵ�һ������Ԫ��֮ǰ�����һ������Ԫ��֮����ÿ��ά�����
            %��replicate���ظ�a��borderԪ��   w+8��h+8
            degraded =convn(gt_extend, rot90(kernel,2), 'valid') ;
            % degraded = convn(gt_extend, rot90(kernel,2), 'valid');
            %%��ת90����ת����90�ȡ�b=rot90��a���Ǿ���A��90����ʱ����ת��
            % ��full��-��Ĭ�ϣ�����������n-d���
            % ����ͬ��-������a��С��ͬ�ľ�������Ĳ��֡�
            % ��valid��-�����ؿ��Լ���Ľ�����֣������ٶ�ʹ����������顣
            noise = randn(size(degraded));
            degraded = degraded + noise * sigma_d;
            figure(11);imshow(gt,[])
            figure(22); imshow(degraded,[]);
            % load network for solver
            % params.net = loadNet(size(gt), use_gpu)
            %% ��ǰ��DAEP
            % params.net = loadNet_qx3channel_diffSigma15([size(gt_extend),3], use_gpu);
            % params.gt = gt;
            %  params2.net = loadNet_qx3channel_diffSigma11([size(gt_extend),3], use_gpu);
            % params2.gt = gt;
            % run DAEP
            %params.num_iter = 500;
            % params.sigma_net = 15;   %20;  %15;  %11;  %
            % params.num_iter = 500;
            % params2.sigma_net = 11;   %20;  %15;  %11;  %
            % params2.num_iter =500;
            %% ���ڵ�DAEP_MWCNN
            % load(['D:\ZJJ_HZN\CNN_v3_noisy_noise�޸�_good5channel\TrainingCodes_CNN_v1.1\data\model_sigma25\DnCNN-epoch-7.mat']); %%% for sigma in [0,55]
            for epoch = [33,40,41]
                load(['E:\ZJJ_HZN\github code\Test code\Image Deblurring\model_sigma25\DnCNN-epoch-' num2str(epoch) '.mat']);
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
                params.sigma_net = 25;
                sigma=params.sigma_net ;
                
                %% ���ڵ�DAEP
                [map_deblur_extend,psnr_ssim,psnr_psnr,prior_err ]= TDAEP_5channel_simga(degraded, kernel, sigma_d, params);
                % map_deblur_extend = DAEP_deblurmulti_mwcnn(degraded, kernel, sigma_d, params, params2,net1, net2,net1_backword,net2_backword);
                map_deblur = map_deblur_extend(pad(1)+1:end-pad(1),pad(2)+1:end-pad(2),:);
                [psnr4, ssim4, fsim4, ergas4, sam4] = MSIQA(gt, map_deblur);
                [psnr4, ssim4, fsim4, ergas4, sam4]
                
                save (['./maxepoch_7m8d_image/',fn1 '_kernels',num2str(aa,'%d'),'_alpha',num2str(a1,'%d'),'_epoch',num2str(epoch,'%d')],'psnr_psnr','psnr_ssim');
                
                % figure;
                % subplot(131);% imshow(gt/255); title('Ground Truth')
                % subplot(132);
                % imshow(degraded/255); title('Blurry')
                % subplot(133);
                % imshow(map_deblur/255); title('Restored')
            end
        end
    end
end