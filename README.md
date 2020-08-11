# Master Thesis 2020
## Evaluation of synergistic effects of tensor decomposition methods within (deep) neural network applications
The master thesis of Tobias Engelhardt Rasmussen, to be performed in the fall of 2020.

Convolutional Neural Networks (CNN) are widely used for image analysis related tasks, e.g. for image classification. A CNN typically involves so called convolutional layers. In brief, a convolutional layer can be understood as an image filter and is essential to extract decisive image features/structures for subsequent image classification and/or prediction. To obtain sufficient prediction accuracy CNNs often contain many layers (= deep learning), resulting in a large number of fitted parameters. The number of fitted parameters also depends on the input size of the data. Hereby, the number of fitted parameters grows exponentially as the image resolution increases. Therefore, training CNNs requires considerable amounts of computational resources and time.  

The project aims at assessing possible synergistic effects between tensor decompositions, such as TUCKER and/or PARAFAC and CNN applications. In this context, the tensor decomposition shall be understood as a data compression method. It shall be applied either prior to training the neural network to reduce the input dimensionality, or within the neural network to "compress the fitted parameters" and to potentially replace convolutional layers. The synergistic effects of the tensorized neural networks (TNN) are meant to lead to increased computational efficiency and simpler network architectures, while at the same time performing at maximal prediction accuracy.

The project will implement tensor decompositions, such as TUCKER and/or PARAFAC, prior/within a chosen CNN architecture. Initially, a simple (image) data set will be simulated to test the methodology under well controlled conditions. In a second step, a suitable data set (from state-of-the-art papers) will be selected to assess and benchmark the developed TNNs against state-of-the-art performance(s).

## Supervisors
- Andreas Baum, DTU
- Line Katrine Harder Clemmensen, DTU

### Period: 31. August 2020 - 31. January 2021
