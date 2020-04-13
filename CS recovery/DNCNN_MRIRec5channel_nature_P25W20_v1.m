function [map,psnr_ssim,psnr_psnr,data_result] = DNCNN_MRIRec5channel_nature_P25W20_v1(Im, degraded, mask, sigma_d, net, params)
% Implements stochastic gradient descent (SGD) maximum-a-posteriori for image deblurring described in:
% S. A. Bigdeli, M. Zwicker, "Image Restoration using Autoencoding Priors".
%实现图像去模糊的随机梯度下降的最大后验
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


if ~any(strcmp('net',fieldnames(params)))
    error('Need a DAE network in params.net!');
end

if ~any(strcmp('num_iter',fieldnames(params)))
    params.num_iter = 600;
end

if ~any(strcmp('gamma',fieldnames(params)))
    params.gamma = 5.875;
end

if ~any(strcmp('mu',fieldnames(params)))
    params.mu = .9;
end

if ~any(strcmp('alpha',fieldnames(params)))
    params.alpha = .1;
end

disp(params)

params.gamma = params.gamma * 4;
pad = [0, 0];
map = padarray(Im, pad, 'replicate', 'both');

% sigma_eta = sqrt(2) * params.sigma_net;
% relative_weight = params.gamma/(sigma_eta^2)/(params.gamma/(sigma_eta^2) + 1/(sigma_d^2));

step = zeros(size(map));
psnr_ssim = zeros(params.num_iter,1);
psnr_psnr = zeros(params.num_iter,1);
data_result = zeros(params.num_iter,256,256);
if any(strcmp('gt',fieldnames(params)))
    psnr = computePSNR(abs(params.gt), abs(map), pad);
    disp(['Initialized with PSNR: ' num2str(psnr)]);
end

for iter = 1:params.num_iter
    if any(strcmp('gt',fieldnames(params)))
        disp(['Running iteration: ' num2str(iter)]);
        tic();
    end
    
    prior_err_sum = zeros(size(map));
    repeat_num = 3;  %3;   %8;  %1; %12;
    for iiii = 1:repeat_num
        if params.modelchannel == 3
            map_all = repmat(map,[1,1,3]);
        else
            map_all = repmat(map,[1,1,1]);   %   %
        end
        % compute prior gradient 1
%         input = abs(map_all);
        input = real(map_all); % Switch channels for caffe    (:,:,[3,2,1])               
        input = input / 255;
        input1 = zeros(size(input,1)+1, size(input,2)+1, 5, 1, 'single');
        addnoise(:,:,1,:) = params.Psigma_net/255*randn(size(input1(:,:,1,:)),'single');
        addnoise(:,:,2:5,:) = params.Wsigma_net/255*randn(size(input1(:,:,2:5,:)),'single');
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
        
        if params.modelchannel == 3
            prior_err1 = mean(prior_err,3) * 255;
        else
            prior_err1 = prior_err * 255;   %
        end
        % compute prior gradient 2
%         input = imag(map_all); % Switch channels for caffe    (:,:,[3,2,1])
%         input = input / 255;
%         input1 = zeros(size(input,1)+1, size(input,2)+1, 5, 1, 'single');
%         addnoise(:,:,1,:) = params.Psigma_net/255*randn(size(input1(:,:,1,:)),'single');
%         addnoise(:,:,2:5,:) = params.Wsigma_net/255*randn(size(input1(:,:,2:5,:)),'single');
%         [ld,hd,lr,hr] = wfilters('bior1.1');
%         input1(2:end,2:end,1) = input;
%         [Wave_S,Wave_HW,Wave_WH,Wave_WW,etl] = ocwt2dliu1(input,ld,hd,1);
%         input1(:,:,2) = Wave_S;input1(:,:,3) = Wave_HW{1,1};input1(:,:,4) = Wave_WH{1,1};input1(:,:,5) = Wave_WW{1,1};
%         input1 = input1 + addnoise;
%         input1 = single(input1);
%         if params.useGPU,   input1 = gpuArray(input1);    end
%         net.eval_forwardback({'input', input1}, addnoise, {'prediction', 1}) ;
%         output_der = gather(squeeze(gather(net.vars(net.getVarIndex('input')).der)));
%         prior_err = zeros(size(input,1), size(input,2), 2, 1, 'single');
%         prior_err(:,:,1) = output_der(2:end,2:end,1);
%         Wave_S = output_der(:,:,2); Wave_HW{1,1} = output_der(:,:,3); Wave_WH{1,1} = output_der(:,:,4); Wave_WW{1,1} = output_der(:,:,5);
%         prior_err(:,:,2) = iocwt2dliu1(Wave_S,Wave_HW,Wave_WH,Wave_WW,etl,lr,hr);
%         prior_err = mean(prior_err,3);
%         if params.modelchannel == 3
%             prior_err2 = mean(prior_err,3) * 255;
%         else
%             prior_err2 = prior_err * 255;   %
%         end
%         if iter > 1;prior_err2 = 0; end
        prior_err_sum = prior_err_sum + prior_err1;% + sqrt(-1)*prior_err2;
    end
    
    for iiii = 1:repeat_num
        if params.modelchannel == 3
            map_all = repmat(map,[1,1,3]);
        else
            map_all = repmat(map,[1,1,1]);   %   %
        end
        % compute prior gradient 1
%         input = abs(map_all);
        input = real(map_all);% Switch channels for caffe    (:,:,[3,2,1])
        input = input / 255;
        input1 = zeros(size(input,1)+1, size(input,2)+1, 5, 1, 'single');
        addnoise(:,:,1,:) = params.Psigma_net/255*randn(size(input1(:,:,1,:)),'single');
        addnoise(:,:,2:5,:) = params.Wsigma_net/255*randn(size(input1(:,:,2:5,:)),'single');
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
        if params.modelchannel == 3
            prior_err1 = mean(prior_err,3) * 255;
        else
            prior_err1 = prior_err * 255;   %
        end
 
%         input = imag(map_all); % Switch channels for caffe    (:,:,[3,2,1])
%         input = input / 255;
%         input1 = zeros(size(input,1)+1, size(input,2)+1, 5, 1, 'single');
%         addnoise(:,:,1,:) = params.Psigma_net/255*randn(size(input1(:,:,1,:)),'single');
%         addnoise(:,:,2:5,:) = params.Wsigma_net/255*randn(size(input1(:,:,2:5,:)),'single');
%         [ld,hd,lr,hr] = wfilters('bior1.1');
%         input1(2:end,2:end,1) = input;
%         [Wave_S,Wave_HW,Wave_WH,Wave_WW,etl] = ocwt2dliu1(input,ld,hd,1);
%         input1(:,:,2) = Wave_S;input1(:,:,3) = Wave_HW{1,1};input1(:,:,4) = Wave_WH{1,1};input1(:,:,5) = Wave_WW{1,1};
%         input1 = input1 + addnoise;
%         input1 = single(input1);
%         if params.useGPU,   input1 = gpuArray(input1);    end
%         net.eval_forwardback({'input', input1}, addnoise, {'prediction', 1}) ;
%         output_der = gather(squeeze(gather(net.vars(net.getVarIndex('input')).der)));
%         prior_err = zeros(size(input,1), size(input,2), 2, 1, 'single');
%         prior_err(:,:,1) = output_der(2:end,2:end,1);
%         Wave_S = output_der(:,:,2); Wave_HW{1,1} = output_der(:,:,3); Wave_WH{1,1} = output_der(:,:,4); Wave_WW{1,1} = output_der(:,:,5);
%         prior_err(:,:,2) = iocwt2dliu1(Wave_S,Wave_HW,Wave_WH,Wave_WW,etl,lr,hr);
%         prior_err = mean(prior_err,3);
%         if params.modelchannel == 3
%             prior_err2 = mean(prior_err,3) * 255;
%         else
%             prior_err2 = prior_err * 255;   %
%         end
%         if iter > 1;prior_err2 = 0; end
        prior_err_sum = prior_err_sum + prior_err1;% + sqrt(-1)*prior_err2;
    end
    
    prior_err = prior_err_sum/repeat_num/2;
    
    % compute data gradient    %     map_conv = convn(map,rot90(kernel,2),'valid');
    data_err = zeros(size(prior_err));   %convn(map_conv-degraded,kernel,'full');
    
    % sum the gradients
    err = prior_err;  %relative_weight*prior_err + (1-relative_weight)*data_err;  %
    
    % update
    step = params.mu * step - params.alpha * err;
    map = map + step;
    
    temp_FFT = fft2(map);
    temp_FFT(mask==1) = degraded(mask==1);  %
    map = ifft2(temp_FFT);
    map = real(map);
%     map = abs(map);
%     if iter == 20; map = real(map); end
%     map = min(255,max(0,map));
%     map = real(map);
%     map = real(map) + 0.01*imag(map);
    [psnr4, ssim4, ~] = MSIQA(abs(params.gt),  abs(map(pad(1)+1:end-pad(1),pad(2)+1:end-pad(2),:)));
    %     psnr_ssim(iter,1)  = iter;
    psnr_psnr(iter,1)  = psnr4;
    psnr_ssim(iter,1)  = ssim4;
    %     map = min(255,max(0,map));
%     if mod(iter,60)==0, figure(200+iter);imshow([abs(map)],[]);end
    data_result(iter,:,:)  = map;
    if any(strcmp('gt',fieldnames(params)))
        psnr = computePSNR(abs(params.gt), abs(map), pad);
        disp(['PSNR is: ' num2str(psnr) ', iteration finished in ' num2str(toc()) ' seconds']);
    end
    
end
