# TDAEP
The Code is created based on the method described in the following paper: Jinjie Zhou, Zhuonan He, Xiaodong Liu , Shanshan Wang , Qiegen Liu , Yuhao Wang.
Transformed Denoising Autoencoder Prior for Image Restoration
# Motivation
Image restoration problem is usually ill-posed, which can be alleviated by learning image prior. Inspired by considerable performance of utilizing priors in pixel domain and wavelet domain jointly, we propose a novel transformed denoising autoencoder as prior (TDAEP). The core idea behind TDAEP is to enhance the classical denoising autoencoder (DAE) via transform domain, which captures complementary information from multiple views. Specifically, 1-level nonorthogonal wavelet coefficients are used to form 4-channel feature images. Moreover, a 5-channel tensor is obtained by stacking the original image under the pixel domain and 4-channel feature images under the wavelet domain. Then we train the transformed DAE (TDAE) with the 5-channel tensor as network input. The optimized image prior is obtained based on the trained autoencoder, and incorporated into an iterative restoration procedure with the aid of the auxiliary variable technique. The resulting model is addressed by proximal gradient descent technique. Numerous experiments demonstrated that TDAEP outperforms a set of image restoration benchmark algorithms.
# Figs
# Table
# Visual Comparisons
# Requirements and Dependencies
MATLAB R2016b
Cuda-9.0
MatConvNet
(https://pan.baidu.com/s/1ZsKlquIHqtgJYlq3iKNsdg Password：p130)
Pretrained Model
(https://pan.baidu.com/s/1Aa22avm0499VWq7kMvuoXA Password：sjuu)

