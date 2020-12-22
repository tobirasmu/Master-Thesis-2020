## Loading the Data

load_THETIS.py
Imports the different videos given a set of hyper-parameters, stacks the videos in a big tensor using appropriate shaping and saves the tensor to a file that can later be accessed more efficiently. Loading the data requires a file for each shot type manually annotated and pre-made. The function "writeNames2file" writes all the file names of a directory to a file in order to carry out the manual annotation.


## General Functions

video_functions.py
Holds many general functions for carrying out different tasks:
 - Learning (training epoch, evaluation epoch, getting variables from cuda and vice versa)
 - Plotting (writing tensor to video, showing frames and plotting accuracies)
 - Decomposing (decomposing every type of layer using tucker)
 - Evaluating (Timing and calculating the number of FLOPs for different types of layers)


video_networks.py
Holds the different network architectures and can be used to initialise a network or decompose it (both for THETIS and VGG-16)


## Training the networks

FH_BH_input_decomp.py
Decomposes the stacked videos and uses the loadings as the input to a very simple network


FH_BH_CNN.py
Trains the original network and saves the trained network to a file


CNN_decomp.py
Decomposes the original network and fine-tunes the compressed network


## Timing the networks

#### layer_timing_functions.py
Holds the classes for single layer architectures to be used for timing a single layer


timing.py
Times the different networks - both full and layer-wise

