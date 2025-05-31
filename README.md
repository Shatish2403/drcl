# MultiClass Classification Of Diabetic Retinopathy using Swin V2.
This Repo contains the Diabetic Retinopathy Classification Done by Modified Swin V2 with Selective Reattention Branch with higher performance than baseline model.

Here we used Different Image Processing Techniques like CLAHE, Adaptive Gamma Correction and normalization etc. 
The Main Work in Here is Modified Swinv2 in which the #Window Attention Module is inherited and overrided.
We Addded an uncertainty branch which Calculating the window Attention, and then it undergoes the MLP and give a Uncertainty Score it is then used in the attention to rescale the Window Attention.

This Model acheives 97% Accuracy on Testing Data from APTOS 2019 which is 10% (approx) higher than the baseline model with same hyper parameters.
