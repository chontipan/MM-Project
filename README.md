# MM-Project
Music Genre Classification
This assignment used small FMA dataset, and classified only 4 genres namely Electronic, Pop, Folk and Instrument.
Each clip also is sliced into 3 pieces of 10 second to increase numbers of samples. So numbers of samples are 12,000 samples and splited into training, validation and testing (ratio: 80%,10%,10%). The input shape is 480 (Timeframe) x 128 (Frequency) for Melspectrogram input.


This project aimed to compare a variety of tagging music genre algorithms including K-Nearest Neighbors, Support Vector, Convolution Neural Networks, Convolution Recurrent Neural Networks (CRNN) and Parallel Convolution Recurrent Neural Networks (PCRNN). 


For CRNN and PCRNN model, I modified some parameters and add few layers from this repository (https://github.com/priya-dwivedi/Music_Genre_Classification)
