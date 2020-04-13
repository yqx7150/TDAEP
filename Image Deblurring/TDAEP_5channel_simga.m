function [map,psnr_ssim,psnr_psnr,prior_err] = TDAEP_5channel_simga(degraded, kernel, sigma_d, params)
% Implements stochastic gradient descent (SGD) maximum-a-posteriori for image deblurring described in:
% S. A. Bigdeli, M. Zwicker, "Image Restoration using Autoencoding Priors".
%
%
% Input:
% degraded: Observed degraded RGB input image in range of [0, 255].
% kernel: Blur kernel (internally flipped for convolution).
% sigma_d: Noise standard deviation.
% params: Set of parameters.
% params.net: The DAE Network object loaded from MatCaffe.
%
% Optional parameters:
% params.sigma_net: The standard deviation of the network training noise. default: 25
% params.num_iter: Specifies number of iterations.
% params.gamma: Indicates the relative weight between the data term and the prior. default: 6.875
% params.mu: The momentum for SGD optimization. default: 0.9
% params.alpha the step length in SGD optimization. default: 0.1
%
%
% Outputs:
% map: Solution.


% if ~any(strcmp('net',fieldnames(params)))
%     error('Need a DAE network in params.net!');
% end

if ~any(strcmp('sigma_net',fieldnames(params)))
    params.sigma_net = 25;
end

if ~any(strcmp('num_iter',fieldnames(params)))
    params.num_iter =400;
end

if ~any(strcmp('gamma',fieldnames(params)))
    params.gamma = 5.875;
end

if ~any(strcmp('mu',fieldnames(params)))
    params.mu = 0.9;
end

if ~any(strcmp('alpha',fieldnames(params)))
    params.alpha = .1;
end

net = params.net;
disp(params)

params.gamma = params.gamma * 4;
pad = floor(size(kernel)/2);
map = padarray(degraded, pad, 'replicate', 'both');

sigma_eta = sqrt(2) * params.sigma_net;
relative_weight = params.gamma/(sigma_eta^2)/(params.gamma/(sigma_eta^2) + 1/(sigma_d^2));

step = zeros(size(map));

if any(strcmp('gt',fieldnames(params)))
    map_center = map(pad(1)+1:end-pad(1),pad(2)+1:end-pad(2),:);
%     psnr = computePSNR(params.gt, map_center, pad);
%     disp(['Initialized with PSNR: ' num2str(psnr)]);
end

psnr_ssim = zeros(params.num_iter,1);
for iter = 1:params.num_iter
    if any(strcmp('gt',fieldnames(params)))
        disp(['Running iteration: ' num2str(iter)]);
        tic();
    end
    
    prior_err_sum = zeros(size(map));
    repeat_num = 3;  %3;   %8;  %1; %12;
    
    for iiii = 1:repeat_num
    % compute prior gradient
    map_all = map;  %repmat(,[1,1,3]);
    input = map_all;  %(:,:,[3,2,1]); % Switch channels for caffe
    input = input / 255;
    input1 = zeros(size(input,1)+1, size(input,2)+1, 5, 1, 'single');
    addnoise = params.sigma_net/255*randn(size(input1),'single');
    [ld,hd,lr,hr] = wfilters('bior1.1');
    input1(2:end,2:end,1) = input;
    [Wave_S,Wave_HW,Wave_WH,Wave_WW,etl] = ocwt2dliu1(input,ld,hd,1);
    input1(:,:,2) = Wave_S;input1(:,:,3) = Wave_HW{1,1};input1(:,:,4) = Wave_WH{1,1};input1(:,:,5) = Wave_WW{1,1};
    input1 = input1 + addnoise;
    input1 = single(input1);
    if params.useGPU,   input1 = gpuArray(input1);    end
    net.eval_forwardback({'input', input1}, addnoise, {'prediction', 1}) ;
    output_der = gather(squeeze(gather(net.vars(net.getVarIndex('input')).der)));
    prior_err = zeros(size(input,1), size(input,2), 2, 1, 'single');
    prior_err(:,:,1) = output_der(2:end,2:end,1);
    Wave_S = output_der(:,:,2); Wave_HW{1,1} = output_der(:,:,3); Wave_WH{1,1} = output_der(:,:,4); Wave_WW{1,1} = output_der(:,:,5);
    prior_err(:,:,2) = iocwt2dliu1(Wave_S,Wave_HW,Wave_WH,Wave_WW,etl,lr,hr);
    prior_err = mean(prior_err,3);
    prior_err = prior_err * 255;   %
    prior_err_sum = prior_err_sum + prior_err;
    end
%% Ô­À´µÄ DAEP
%     for iiii = 1:repeat_num
%     compute prior gradient
%     map_all = repmat(map,[1,1,3]);
%     input = map_all(:,:,[3,2,1]); % Switch channels for caffe    
%     noise = randn(size(input)) * params2.sigma_net;
%     rec = params2.net.forward({input+noise});
%     figure(200+iter);imshow(uint8(rec{1,1}),[])
%     
%     prior_err = input - rec{1};
%     rec = params2.net.backward({-prior_err});
%     prior_err = prior_err + rec{1};
% 
%     tmp = prior_err;
%     prior_err = prior_err*0;
%     prior_err(pad(1)+1:end-pad(1), pad(2)+1:end-pad(2),:) = tmp(pad(1)+1:end-pad(1), pad(2)+1:end-pad(2),:);
%     prior_err = prior_err(:,:,[3,2,1]);
%      figure(4); imshow(prior_err(:,:,1),[]);
%     figure(5); imshow(prior_err(:,:,2),[]);
%     figure(6); imshow(prior_err(:,:,3),[])
%      prior_err = mean(prior_err,3); 
%     prior_err_sum = prior_err_sum + prior_err;
%      figure(2);imshow(prior_err,[])
%     end
 for iiii = 1:repeat_num
    % compute prior gradient
    map_all = map;  %repmat(,[1,1,3]);
    input = map_all;  %(:,:,[3,2,1]); % Switch channels for caffe
    input = input / 255;
    input1 = zeros(size(input,1)+1, size(input,2)+1, 5, 1, 'single');
    addnoise = params.sigma_net/255*randn(size(input1),'single');
    [ld,hd,lr,hr] = wfilters('bior1.1');
    input1(2:end,2:end,1) = input;
    [Wave_S,Wave_HW,Wave_WH,Wave_WW,etl] = ocwt2dliu1(input,ld,hd,1);
    input1(:,:,2) = Wave_S;input1(:,:,3) = Wave_HW{1,1};input1(:,:,4) = Wave_WH{1,1};input1(:,:,5) = Wave_WW{1,1};
    input1 = input1 + addnoise;
    input1 = single(input1);
    if params.useGPU,   input1 = gpuArray(input1);    end
    net.eval_forwardback({'input', input1}, addnoise, {'prediction', 1}) ;
    output_der = gather(squeeze(gather(net.vars(net.getVarIndex('input')).der)));
    prior_err = zeros(size(input,1), size(input,2), 2, 1, 'single');
    prior_err(:,:,1) = output_der(2:end,2:end,1);
    Wave_S = output_der(:,:,2); Wave_HW{1,1} = output_der(:,:,3); Wave_WH{1,1} = output_der(:,:,4); Wave_WW{1,1} = output_der(:,:,5);
    prior_err(:,:,2) = iocwt2dliu1(Wave_S,Wave_HW,Wave_WH,Wave_WW,etl,lr,hr);
    prior_err = mean(prior_err,3);
    prior_err = prior_err * 255;   %
    prior_err_sum = prior_err_sum + prior_err;    
 end
    prior_err = prior_err_sum/repeat_num/2;
%     figure(333);imshow(prior_err,[])
    % compute data gradient
    map_conv = convn(map,rot90(kernel,2),'valid');
    data_err = convn(map_conv-degraded,kernel,'full');
    
    % sum the gradients
    err = relative_weight*prior_err + (1-relative_weight)*data_err;
   
    % update
    step = params.mu * step - params.alpha * err;
    map = map + step;
    map = min(255,max(0,map));
%     figure(444); imshow(map,[]);
%     if mod(iter,20)==0, figure(200+iter);imshow([uint8(map_center)],[]);end

    [psnr4, ssim4, ~] = MSIQA(params.gt,  map(pad(1)+1:end-pad(1),pad(2)+1:end-pad(2),:));
%     psnr_ssim(iter,1)  = iter;
    psnr_psnr(iter,1)  = psnr4;
    psnr_ssim(iter,1)  = ssim4;
    
    
    if any(strcmp('gt',fieldnames(params)))
        map_center = map(pad(1)+1:end-pad(1),pad(2)+1:end-pad(2),:);
        psnr = csnr(params.gt, map_center, 0,0);
         ssim=cal_ssim(params.gt/255, map_center/255, 0,0);   %ssim=cal_ssim(params.gt, map_center, 0,0);
        disp(['PSNR is: ' num2str(psnr4),'/','SSIM is: ' num2str(ssim4)   ', iteration finished in ' num2str(toc()) ' seconds']);
    end
 end
end