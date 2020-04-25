EasyLab
=
工欲善其事，必先利其器 Sharp tools make good work.
-

This is a high level wrapper of several hot machine learning and deep learning python library. We hope this wrapper can 
help beginner researcher to build experiment quickly instead of complicated environment setting(distributed training, 
fp16 training, GPU acceleration). The backend engine is accelerated by some fancy library like Pytorch,
Nvidia apex, Ctupy. etc. Additionally, you can find some experiment reproducing and benchmark scripts implemented by 
us. 
Limit to my experience, my implementation might not achieve others'. So these scripts just suitable for beginner. I will 
keep updating this repository in the future, not only expand the functions, but also accelerate the program as possible 
as I can. If you have advise on some functions or find some bugs, feel free to leave a message here, I am very appreciate 
that. Thanks for your reading.

Contents
-

* easylab
    * dl
        * dataset
        * model
        * lossfunction
        * network
        * metrics
        * utils
        * tools
        
    * ml
    
    * examples
        * dl
            * recognition
            * detection
            * segmentation
            * gan
            * etc
        * ml
    
    
    
Requirements    
-
Author's environment. `Update Date: 09/06/2019`

- [Anaconda](https://anaconda.org/) (Python 3.6.5)
- [Pytorch](https://pytorch.org/) (1.1.0)
- [Nvidia APEX](https://github.com/NVIDIA/apex) (0.1)
- CUDA10.0 and cudnn7


Quick Start
-
Linux

    $ git clone https://github.com/xyzacademic/easylab.git
    
Community Members
-
EasyLab is currently maintained by [Yunzhe Xue](https://github.com/xyzacademic)