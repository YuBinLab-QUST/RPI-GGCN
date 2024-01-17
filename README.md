# RPI-GGCN
RPI-GGCN: Prediction of RNA-protein interaction based on interpretability gated graph convolution neural network and co-regularized variational autoencoders
###PRPI-GGCN uses the following dependencies:

 * Python 3.9
 * numpy
 * scipy
 * scikit-learn
 * pandas
 * tensorflow 
 * keras

###Guiding principles: 

**The dataset file contains seven datasets, among which RPI369, RPI488, RPI1446, RPI1807, RPI2241, NPInter v3.0.

**Feature extraction：
 * feature-RNA is the implementation of k-mer, dinucleotide cross-covariance, KGap descriptor, and pseudo-trinucleotide composition for RNA.
 * feature-protein is the implementation of triplex combinatorial information coding, grouped dipeptide composition, QSOder descriptor, expected mean dipeptide deviation, and amino acid composition for protein.
 

**Feature_selection:
 * ALL_select is the implementation of all feature selection methods used in this work, among which LLE,LASSO,SE,MDS,LR,OMP,TSVD,GINI,LGBM,IG,MRMD,Co-VAE.

**Classifier:
 * RPI-GGCN_model.py is the implementation of our model in this work.
 * classical classifier is the implementation of classical classifierS compared in this work, among which RF,SVM,MLP,ET,AdaBoost,KNN.
 * deep learning classifier is the implementation of deep learning classifiers compared in this work, among which  CNN, DNN, GRU,RNN,GAN,GNN,GCN.

