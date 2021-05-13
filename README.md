# Object-centric-Image-recognition-DL

Tested on Cifar10, Caltech101 and Caltech256 datasets.

# Methodology:
![methodology](https://github.com/BirvaPatel/Object-centric-Image-recognition-DL/Methodology.PNG)
As shown above, We used augmented data for feature extraction into two pretrained model: one is Resnet101 and another is InceptionV3. Then, we predict the data from it and use predicted data in another pretrained model (DenseNet201) for the classification task and evaluate the model.


