ReconWithKSAE

See the TMI paper "Iterative low-dose CT reconstruction with priors trained by nerual networks", by Dufan Wu, Kyungsang Kim, Geoges El Fakhri, and Quanzheng Li, 2017. 

The code uploaded here is only for the network training and deploying. The core GPU code for the image reconstruction (forward and backward model) was not included, because I am not the sole author for that piece of code. However, the reconstruction python code should give enough explanation on how to integrate the neural network into the reconstruction algorithm (SQS in our case).

The KSAE is an unsurpervised learning algorithm, please keep this in mind when doing comparison studies with other neural network based methods like ADMM-net. 
