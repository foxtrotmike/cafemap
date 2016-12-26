Python Implementation of CAFÉ-Map: Context Aware Feature Mapping for mining high dimensional biomedical data 
as described in paper [1] 

Authors: 

Dr. Fayyaz Minhas (afsar <at> pieas dot edu dot pk)
Amina Asif  (a.asif.shah01 <at> gmail dot com )
Muhammad Arif (syedmarif2003 <at> yahoo dot com)
Downloaded From: http://faculty.pieas.edu.pk/fayyaz/software.html#cafemap

This folder contains the package "cafeMap" and all example files. 

INSTALLATION INSTRUCTIONS FOR THE PACKAGE:
1. Go to directory ..../cafemap-master/cafeMap in command prompt
2. Execute the command: pip install .
or
python setup.py install


CafeMap package has the following modules:

cafemap.py: class implementation of cafemap according to the algorithm presented in the paper

instance.py: contains the definition of an instance to be used by cafemap

cv.py: parallel implementation of cross validation methods is present in this file.

llc.py: This module implements the approximate Locality Constrained Linear Coding as described in the 2010 paper 
by Wang et al. [2]. Given array of datapoints X (N x d) and codebook C (c x d), it returns a vector of approximated 
points Y = G * C. LLC introduces sparsity by forcing those coefficients of a given data point that correspond to codebook 
vectors which are not that point's k-nearest neighbors. LLC also uses regularization. 

utils.py: contains utility functions to facilitate compilation of results



Following files contain the code that generated the results published in the study:

results_table1.py produces the results presented in Table 1 of [1]
l-shaped.py: produces the plots presented in Figure 2 of [1]
2x2checker.py: produces the plots presented in Figure 3 of [1]
toy_circle.py: produces the plots for circular data as presented in Figure 4 of [1]. 
arcene.py: produces the plots presented in Figure 6 of [1]
prostate.py: produces the clustering analysis plots presented in Figure 8 of [1]

Note: the plots may vary from the published according to the selection of parameters while running the code.

Example.py illustrates the use of cafeMap package. To use parallel version of the training method, joblib module should be 
installed. All the parameters are explained in comments.  



References:
[1]F. ul A. A. Minhas, A. Asif, and M. Arif, “CAFÉ-Map: Context Aware Feature Mapping 
for mining high dimensional biomedical data,” Computers in Biology and Medicine, vol. 79, pp. 68–79, Dec. 2016.

[2] Wang, Jinjun, Jianchao Yang, Kai Yu, Fengjun Lv, T. Huang, and Yihong Gong. 
“Locality-Constrained Linear Coding for Image Classification.” In 2010 IEEE Conference on Computer Vision and 
Pattern Recognition (CVPR), 3360–67, 2010. doi:10.1109/CVPR.2010.5540018.

Acknowledgments: We used the ROC module implemented by Dr. Asa Ben-Hur and Mike Hamilton which follows:
Theoretical and pratical concepts from 
Fawcett, T.  ROC graphs: Notes and pratical considerations
for data mining researchers.  HPL-2003-4, 2003.
