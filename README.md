# TDAEP
The Code is created based on the method described in the following paper: Jinjie Zhou, Zhuonan He, Xiaodong Liu , Shanshan Wang , Qiegen Liu , Yuhao Wang.
Transformed Denoising Autoencoder Prior for Image Restoration
# Motivation
Image restoration problem is usually ill-posed, which can be alleviated by learning image prior. Inspired by considerable performance of utilizing priors in pixel domain and wavelet domain jointly, we propose a novel transformed denoising autoencoder as prior (TDAEP). The core idea behind TDAEP is to enhance the classical denoising autoencoder (DAE) via transform domain, which captures complementary information from multiple views. Specifically, 1-level nonorthogonal wavelet coefficients are used to form 4-channel feature images. Moreover, a 5-channel tensor is obtained by stacking the original image under the pixel domain and 4-channel feature images under the wavelet domain. Then we train the transformed DAE (TDAE) with the 5-channel tensor as network input. The optimized image prior is obtained based on the trained autoencoder, and incorporated into an iterative restoration procedure with the aid of the auxiliary variable technique. The resulting model is addressed by proximal gradient descent technique. Numerous experiments demonstrated that TDAEP outperforms a set of image restoration benchmark algorithms.
# Figs
![repeat-MWDMSP](https://github.com/yqx7150/MWDMSP/blob/master/test%20code/utils/flowchart.png)
Fig. 1. Flow-chart of the network used in MWDMSP.
![repeat-MWDMSP](https://github.com/yqx7150/MWDMSP/blob/master/test%20code/utils/flowchart.png)
Fig. 1. Flow-chart of the network used in MWDMSP.
![repeat-MWDMSP](https://github.com/yqx7150/MWDMSP/blob/master/test%20code/utils/flowchart.png)
Fig. 1. Flow-chart of the network used in MWDMSP.
# Table
# Visual Comparisons
# Requirements and Dependencies
  MATLAB R2016b
  Cuda-9.0
  MatConvNet
  (https://pan.baidu.com/s/1ZsKlquIHqtgJYlq3iKNsdg Password：p130)
  Pretrained Model——Image Deblurring
  (https://pan.baidu.com/s/1Aa22avm0499VWq7kMvuoXA Password：sjuu)
  Pretrained Mode2——CS Recovery
  (https://pan.baidu.com/s/1Aa22avm0499VWq7kMvuoXA Password：sjuu)
  
## Image Deblurring
'./DemoMRI/demo_MRI.m' is the demo of HF-DAEP for MRI reconstruction.
## CT CS Recovery
'./DemoCT/demo_CTRec.m' is the demo of HF-DAEP for CT reconstruction.
'./DemoCT/ultilies/generateSystemMatrix.m' is used to generate the system matrix.
## Other Related Projects
  * Multi-Channel and Multi-Model-Based Autoencoding Prior for Grayscale Image Restoration  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8782831)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/MEDAEP)   [<font size=5>**[Slide]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)

  * Highly Undersampled Magnetic Resonance Imaging Reconstruction using Autoencoding Priors  
[<font size=5>**[Paper]**</font>](https://cardiacmr.hms.harvard.edu/files/cardiacmr/files/liu2019.pdf)  [<font size=5>**[Code]**</font>](https://github.com/yqx7150/EDAEPRec)   [<font size=5>**[Slide]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)

  * Learning Priors in High-frequency Domain for Inverse Imaging Reconstruction  
[<font size=5>**[Paper]**</font>](https://arxiv.org/ftp/arxiv/papers/1910/1910.11148.pdf)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/HFDAEP)
 
  * Learning Multi-Denoising Autoencoding Priors for Image Super-Resolution  
[<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/pii/S1047320318302700)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/MDAEP-SR)


